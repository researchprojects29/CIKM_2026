import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS_JSON = HERE / "solution-final-report-mistral" / "solution-final-output-mistral-for-compare.json"
DEFAULT_ANSWERS_CSV = HERE.parent / "answers.csv"
DEFAULT_OUT_DIR = HERE / "comparison_report-mistral"

# 20 canonical case labels used across the project (mirrors `DATASET/evaluate.py`).
EXPECTED_CASE_LABELS: List[str] = [
    "full_text + full_diagram",
    "half_text + half_diagram",
    "noisy_text[numeric] + full_diagram",
    "noisy_text[punctuation] + full_diagram",
    "noisy_text[sentence] + full_diagram",
    "noisy_text[typo] + full_diagram",
    "full_text + noisy_diagram[bg_change]",
    "full_text + noisy_diagram[blur]",
    "full_text + noisy_diagram[illumination]",
    "full_text + noisy_diagram[irr_objects]",
    "full_text + noisy_diagram[pixel_noise]",
    "noisy_half_text[numeric] + half_diagram",
    "noisy_half_text[punctuation] + half_diagram",
    "noisy_half_text[sentence] + half_diagram",
    "noisy_half_text[typo] + half_diagram",
    "half_text + noisy_half_diagram[bg_change]",
    "half_text + noisy_half_diagram[blur]",
    "half_text + noisy_half_diagram[illumination]",
    "half_text + noisy_half_diagram[irr_objects]",
    "half_text + noisy_half_diagram[pixel_noise]",
]


def load_answers_csv(path: Path) -> Dict[int, str]:
    answers: Dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            idx_raw = (row.get("problem_index") or "").strip()
            ans_raw = (row.get("answer") or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                continue
            answers[idx] = ans_raw
    return answers


# --- Answer comparison (best-effort, robust to units/latex) ---------------------


def _strip_latex_markup(s: str) -> str:
    s = str(s).strip()
    s = s.replace("$", "")
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = re.sub(r"\\left[\(\[]", "(", s)
    s = re.sub(r"\\right[\)\]]", ")", s)
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    # Remove leading "x = " style prefixes.
    s = re.sub(r"^[A-Za-z][A-Za-z\s]*=\s*", "", s)
    s = re.sub(r"\\,", " ", s)
    s = s.replace("π", r"\pi")
    s = re.sub(r"√\s*\{([^{}]+)\}", r"\\sqrt{\1}", s)
    s = re.sub(r"√\s*([A-Za-z0-9\.]+)", r"\\sqrt{\1}", s)
    # Remove common length units at end (keep powers if present in body).
    s = re.sub(
        r"\s*(cm|mm|km|m|meter|meters)\s*(\^\s*\{?\s*\d+\s*\}?)?\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _extract_plain_number_with_optional_unit(s: str) -> Optional[float]:
    s = _strip_latex_markup(str(s)).strip().lower()
    s = s.rstrip("°").strip()
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*([a-z\u4e00-\u9fff]+)?", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_math_sympy(s: str):
    """
    Returns a Sympy expression/tuple/equality or None.
    If sympy isn't installed, returns None.
    """
    try:
        from sympy.parsing.latex import parse_latex  # type: ignore
        from sympy import sympify  # type: ignore
    except Exception:
        return None

    s = _strip_latex_markup(s)

    tuple_match = re.match(r"^\((.+)\)$", s)
    if tuple_match or ("," in s and not s.startswith("\\")):
        inner = tuple_match.group(1) if tuple_match else s.strip("()")
        parts = [p.strip() for p in inner.split(",")]
        parsed = []
        for p in parts:
            try:
                parsed.append(parse_latex(p))
            except Exception:
                try:
                    parsed.append(sympify(p))
                except Exception:
                    return None
        return tuple(parsed)

    try:
        return parse_latex(s)
    except Exception:
        pass

    try:
        s2 = re.sub(r"(\d)(\s*)([a-zA-Z])", r"\1*\3", s)
        return sympify(s2)
    except Exception:
        return None


def _numeric_equal_sympy(a, b, tol: float = 0.01) -> bool:
    try:
        from sympy import N, Abs  # type: ignore
    except Exception:
        return False
    try:
        return float(N(Abs(a - b))) < tol
    except Exception:
        return False


def _tuple_equal_sympy(a: tuple, b: tuple, tol: float = 0.1) -> bool:
    if len(a) != len(b):
        return False
    return all(_numeric_equal_sympy(ai, bi, tol) for ai, bi in zip(a, b))


def is_correct(predicted: str, ground_truth: str) -> bool:
    """
    Best-effort equivalence:
    - case/whitespace/degree symbol tolerant
    - numeric tolerant
    - sympy-based latex/math equivalence when available
    - substring fallback (GT contained in prediction)
    """
    p = (predicted or "").strip()
    g = (ground_truth or "").strip()
    if not p or not g:
        return False

    p_str = p.strip().lower().rstrip("°").strip()
    g_str = g.strip().lower().rstrip("°").strip()
    if p_str == g_str:
        return True

    p_num = _extract_plain_number_with_optional_unit(p)
    g_num = _extract_plain_number_with_optional_unit(g)
    if p_num is not None and g_num is not None and abs(p_num - g_num) < 0.1:
        return True

    p_sym = _parse_math_sympy(p)
    g_sym = _parse_math_sympy(g)
    if p_sym is None or g_sym is None:
        # fallback: allow extra words like "degrees" or "cm"
        return g_str in p_str

    # Avoid importing sympy classes if sympy isn't available
    try:
        from sympy import Equality, simplify  # type: ignore
    except Exception:
        return g_str in p_str

    if isinstance(p_sym, tuple) and isinstance(g_sym, tuple):
        return _tuple_equal_sympy(p_sym, g_sym)

    if isinstance(p_sym, Equality) and isinstance(g_sym, Equality):
        try:
            diff = simplify(p_sym.lhs - p_sym.rhs - (g_sym.lhs - g_sym.rhs))
            return diff == 0
        except Exception:
            return str(p_sym) == str(g_sym)

    if not isinstance(p_sym, tuple) and not isinstance(g_sym, tuple):
        return _numeric_equal_sympy(p_sym, g_sym)

    return g_str in p_str


def _case_label_from_source_csv(source_csv: str) -> str:
    if not source_csv:
        return "UNKNOWN_CASE"
    # handles "first run\\full_text + full_diagram.csv"
    last = re.split(r"[\\/]", source_csv)[-1]
    if last.lower().endswith(".csv"):
        last = last[:-4]
    return last.strip() or "UNKNOWN_CASE"


@dataclass(frozen=True)
class Row:
    run_folder: str
    case_label: str
    problem_index: int
    ground_truth: str
    predicted: str
    correct: bool
    status: str


def _safe_get(d: Dict[str, Any], *path: str, default: str = "") -> str:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else str(cur)


def load_predictions(path: Path, answers: Dict[int, str]) -> List[Row]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")

    rows: List[Row] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        run_folder = str(item.get("run_folder") or "").strip() or "UNKNOWN_RUN"
        case_label = _case_label_from_source_csv(str(item.get("source_csv") or ""))
        try:
            idx = int(item.get("problem_index"))
        except Exception:
            continue
        gt = str(answers.get(idx, "")).strip()
        pred = _safe_get(item, "model_output", "ANSWER", default="").strip()
        status = str(item.get("status") or "").strip()
        correct = bool(gt) and is_correct(pred, gt)
        rows.append(
            Row(
                run_folder=run_folder,
                case_label=case_label,
                problem_index=idx,
                ground_truth=gt,
                predicted=pred,
                correct=correct,
                status=status,
            )
        )
    return rows


def _pct(n: int, d: int) -> float:
    return 0.0 if d <= 0 else (100.0 * n / d)


def summarize(rows: Iterable[Row]) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    correct = sum(1 for r in rows if r.correct)
    wrong = total - correct
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy_pct": round(_pct(correct, total), 3),
        "wrong_pct": round(_pct(wrong, total), 3),
    }


def group_summaries(rows: List[Row]) -> Dict[str, Any]:
    by_run: Dict[str, List[Row]] = defaultdict(list)
    by_run_case: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
    by_case_all_runs: Dict[str, List[Row]] = defaultdict(list)

    for r in rows:
        by_run[r.run_folder].append(r)
        by_run_case[(r.run_folder, r.case_label)].append(r)
        by_case_all_runs[r.case_label].append(r)

    run_summary = {run: summarize(rs) for run, rs in sorted(by_run.items())}

    run_case_summary: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (run, case), rs in sorted(by_run_case.items()):
        run_case_summary[run][case] = summarize(rs)

    case_summary_all_runs = {case: summarize(rs) for case, rs in sorted(by_case_all_runs.items())}

    # Ensure all 20 cases are present in each run + overall combined case view,
    # even if a case is missing from the consolidated predictions file.
    for run in run_summary.keys():
        for case in EXPECTED_CASE_LABELS:
            if case not in run_case_summary.get(run, {}):
                run_case_summary[run][case] = summarize([])

    for case in EXPECTED_CASE_LABELS:
        if case not in case_summary_all_runs:
            case_summary_all_runs[case] = summarize([])

    return {
        "overall": summarize(rows),
        "by_run": run_summary,
        "by_run_case": run_case_summary,
        "by_case_all_runs": case_summary_all_runs,
    }


def write_detailed_csv(rows: List[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_folder",
                "case_label",
                "problem_index",
                "ground_truth",
                "predicted",
                "correct",
                "status",
            ]
        )
        for r in sorted(rows, key=lambda x: (x.run_folder, x.case_label, x.problem_index)):
            w.writerow(
                [
                    r.run_folder,
                    r.case_label,
                    r.problem_index,
                    r.ground_truth,
                    r.predicted,
                    int(r.correct),
                    r.status,
                ]
            )


def write_summary_json(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(summary: Dict[str, Any], out_path: Path) -> None:
    """
    Flat, spreadsheet-friendly summary:
      - RUN rows (per run)
      - RUN_CASE rows (per run x 20 cases)
      - CASE_ALL_RUNS rows (per case across runs)
      - OVERALL row
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def row(kind: str, run: str, case: str, s: Dict[str, Any]) -> List[Any]:
        return [
            kind,
            run,
            case,
            s["total"],
            s["correct"],
            s["wrong"],
            s["accuracy_pct"],
            s["wrong_pct"],
        ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "run_folder", "case_label", "total", "correct", "wrong", "accuracy_pct", "wrong_pct"])

        # OVERALL
        w.writerow(row("OVERALL", "", "", summary["overall"]))

        # RUN
        for run, s in summary.get("by_run", {}).items():
            w.writerow(row("RUN", run, "", s))

        # RUN_CASE (force canonical order)
        by_run_case = summary.get("by_run_case", {})
        for run in summary.get("by_run", {}).keys():
            for case in EXPECTED_CASE_LABELS:
                s = by_run_case.get(run, {}).get(case, summarize([]))
                w.writerow(row("RUN_CASE", run, case, s))

        # CASE_ALL_RUNS (canonical order)
        by_case = summary.get("by_case_all_runs", {})
        for case in EXPECTED_CASE_LABELS:
            w.writerow(row("CASE_ALL_RUNS", "", case, by_case.get(case, summarize([]))))


def main() -> int:
    predictions_json = Path(os.environ.get("PREDICTIONS_JSON", str(DEFAULT_PREDICTIONS_JSON)))
    answers_csv = Path(os.environ.get("ANSWERS_CSV", str(DEFAULT_ANSWERS_CSV)))
    out_dir = Path(os.environ.get("OUT_DIR", str(DEFAULT_OUT_DIR)))

    if not predictions_json.is_file():
        raise FileNotFoundError(f"Predictions JSON not found: {predictions_json}")
    if not answers_csv.is_file():
        raise FileNotFoundError(f"Answers CSV not found: {answers_csv}")

    answers = load_answers_csv(answers_csv)
    rows = load_predictions(predictions_json, answers)

    summary = group_summaries(rows)

    write_detailed_csv(rows, out_dir / "detailed.csv")
    write_summary_json(summary, out_dir / "summary.json")
    write_summary_csv(summary, out_dir / "summary.csv")

    # Print console-friendly overview
    overall = summary["overall"]
    print("\n=== Overall ===")
    print(
        f"Total={overall['total']} | Correct={overall['correct']} ({overall['accuracy_pct']}%) | "
        f"Wrong={overall['wrong']} ({overall['wrong_pct']}%)"
    )

    print("\n=== By run ===")
    for run, s in summary["by_run"].items():
        print(
            f"{run}: Total={s['total']} | Correct={s['correct']} ({s['accuracy_pct']}%) | "
            f"Wrong={s['wrong']} ({s['wrong_pct']}%)"
        )

    print(
        f"\nWrote:\n"
        f"  {out_dir / 'summary.json'}\n"
        f"  {out_dir / 'summary.csv'}\n"
        f"  {out_dir / 'detailed.csv'}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

