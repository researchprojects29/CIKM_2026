import csv
import json
import os
import re
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# -------------------------------------------------------------------
# Model provider selection (comment/uncomment exactly one)
# -------------------------------------------------------------------
# MODEL_PROVIDER = "ollama"
MODEL_PROVIDER = "openrouter"

# Ollama model
OLLAMA_MODEL = "gemma3:4b"

# OpenRouter model (Sonnet 4.5)
# OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
OPENROUTER_MODEL = "meta-llama/llama-4-scout"
# OPENROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
# Set your API key in environment variable OPENROUTER_API_KEY.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

MAX_RETRIES = 5
RETRY_DELAY = 3
RETRY_BACKOFF = 2
REQUEST_TIMEOUT_SECONDS = 180

BASE_DIR = Path(__file__).resolve().parent
QUESTION_CSV = BASE_DIR.parent / "question-1.csv"
OUTPUT_JSON = BASE_DIR / "solution-final-output-mistral.json"
OUTPUT_JSON_FLAT = BASE_DIR / "solution-final-output-mistral-flat.json"
REPORT_DIR = BASE_DIR / "solution-final-report-mistral"
COMPARE_INPUT_JSON = REPORT_DIR / "solution-final-output-mistral-for-compare.json"

# -------------------------------------------------------------------
# Optional in-code filters.
# -------------------------------------------------------------------
# Folder selection (relative to FINAL OUTPUT). Leave empty to scan all folders.
# Example: RUN_FOLDERS = ["run-1-llama", "run-2-llama", "run-3-llama"]
RUN_FOLDERS = ["run-1-mistral", "run-2-mistral", "run-3-mistral"]

TARGET_INDICES = []

# Case selection format matches triples-local.py exactly.
# Uncomment the case(s) you want to run.
CASES = [
    # Baseline
    (1, "full_text + full_diagram", "./Dataset/Setting_1&2/FTFD"),
    (11, "half_text + half_diagram", "./Dataset/Setting_1&2/Half Text Half Diagram Images"),

    # Noisy text + full diagram (4 text noise types)
    (2, "noisy_text[numeric] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Num Distracto_Q Full Image"),
    (3, "noisy_text[punctuation] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Punctuation Swap_Q Full Image"),
    (4, "noisy_text[sentence] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Sent Distractor_Q Full Image"),
    (5, "noisy_text[typo] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Typo_Q Full Image"),

    # Full text + noisy diagram (5 diagram noise types)
    (6, "full_text + noisy_diagram[bg_change]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/background_noise_fullQ"),
    (7, "full_text + noisy_diagram[blur]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/blur_noise_fullQ"),
    (8, "full_text + noisy_diagram[illumination]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/illumination_noise_fullQ"),
    (9, "full_text + noisy_diagram[irr_objects]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/irrelevant_objects_noise_fullQ"),
    (10, "full_text + noisy_diagram[pixel_noise]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/pixel_noise_fullQ"),

    # Noisy half text + half diagram (4 text noise types)
    (12, "noisy_half_text[numeric] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Numeric_Distractor"),
    (13, "noisy_half_text[punctuation] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Punctuation_Noise_Text"),
    (14, "noisy_half_text[sentence] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Distractor_Noise_Text"),
    (15, "noisy_half_text[typo] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Typo_Noise"),

    # Half text + noisy half diagram (5 diagram noise types)
    (16, "half_text + noisy_half_diagram[bg_change]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/background_noise"),
    (17, "half_text + noisy_half_diagram[blur]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/blur_noise"),
    (18, "half_text + noisy_half_diagram[illumination]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/illumination_noise"),
    (19, "half_text + noisy_half_diagram[irr_objects]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/irrelevant_object_noise"),
    (20, "half_text + noisy_half_diagram[pixel_noise]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/pixel_noise"),
]


SYSTEM_PROMPT = """
You will receive:
1) A question
2) A list of triples in the exact format: (subject, predicate, object)

Use only the provided triples to answer the question.

Return STRICT JSON only in this exact key order:
{
  "ANSWER": "...",
  "REASONING": "..."
}

Rules:
- Do not output markdown.
- Do not output extra keys.
- ANSWER must be concise.
- REASONING should justify the answer step by step from the triples."""


def check_ollama_running():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return None


def validate_provider_config():
    provider = (MODEL_PROVIDER or "").strip().lower()
    if provider not in {"ollama", "openrouter"}:
        raise SystemExit("MODEL_PROVIDER must be either 'ollama' or 'openrouter'.")

    if provider == "openrouter" and not OPENROUTER_API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY is empty. Set environment variable OPENROUTER_API_KEY to use OpenRouter."
        )


def read_questions(path: Path):
    questions = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_idx = str(row.get("problem_index", "")).strip()
            if not raw_idx:
                continue
            try:
                idx = int(raw_idx)
            except ValueError:
                continue

            # Keep even empty questions; caller can decide to skip.
            question_text = str(row.get("question", "")).strip()
            questions[idx] = question_text
    return questions


def collect_run_csvs(final_output_dir: Path):
    csv_paths = []
    for csv_path in sorted(final_output_dir.rglob("*.csv")):
        if csv_path.name.lower() == "question-1.csv":
            continue
        csv_paths.append(csv_path)
    return csv_paths


def normalize_case_name(name: str):
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def read_triples_grouped_by_index(csv_path: Path):
    grouped = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"problem_index", "subject", "predicate", "object"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            return grouped

        for row in reader:
            raw_idx = str(row.get("problem_index", "")).strip()
            if not raw_idx:
                continue
            try:
                idx = int(raw_idx)
            except ValueError:
                continue

            subject = str(row.get("subject", "")).strip()
            predicate = str(row.get("predicate", "")).strip()
            obj = str(row.get("object", "")).strip()
            if not (subject and predicate and obj):
                continue

            triple = f"({subject}, {predicate}, {obj})"
            grouped.setdefault(idx, []).append(triple)
    return grouped


def extract_json_from_text(raw_text: str):
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Prefer first fenced/embedded JSON object if extra prose is present.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        candidate = match.group(0)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Repair common LLM JSON issues:
    # - single quotes instead of double quotes
    # - trailing commas before } or ]
    try:
        candidate = match.group(0).strip()
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = re.sub(r"(?<!\\)'", '"', candidate)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def query_with_ollama(question: str, triples):
    triples_block = "\n".join(triples)
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"TRIPLES:\n{triples_block}\n\n"
        "Return JSON with keys ANSWER and REASONING."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"num_predict": 1200, "temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
            content = (result.get("message") or {}).get("content", "").strip()
            if not content:
                raise ValueError("Empty model response")
            return content
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Model call failed after {MAX_RETRIES} attempts: {exc}") from exc
            time.sleep(delay)
            delay *= RETRY_BACKOFF


def query_with_openrouter(question: str, triples):
    triples_block = "\n".join(triples)
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"TRIPLES:\n{triples_block}\n\n"
        "Return JSON with keys ANSWER and REASONING."
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
            content = (
                ((result.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
            )
            if not content:
                raise ValueError(f"Empty model response. Raw: {result}")
            return content
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Model call failed after {MAX_RETRIES} attempts: {exc}") from exc
            time.sleep(delay)
            delay *= RETRY_BACKOFF


def query_model(question: str, triples):
    provider = (MODEL_PROVIDER or "").strip().lower()
    if provider == "openrouter":
        return query_with_openrouter(question, triples)
    return query_with_ollama(question, triples)


def query_json_fix_only(previous_raw: str):
    """
    Ask model to convert previous response into strict JSON only.
    This is cheaper/faster than a full re-solve and helps reduce INVALID_JSON.
    """
    user_prompt = (
        "Convert the following assistant output into STRICT valid JSON with keys "
        'exactly "ANSWER" and "REASONING".\n'
        "Return ONLY JSON. No markdown. No extra text.\n\n"
        f"TEXT:\n{previous_raw}"
    )
    return query_model(user_prompt, [])


def sanitize_filename(name):
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "unnamed_case"


def save_json(data, filepath: Path):
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_case_json(run_dir: Path, case_label: str, case_records):
    file_stem = sanitize_filename(case_label)
    out_path = run_dir / f"{file_stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(case_records, out_path)
    return out_path


def print_summary(records):
    case_stats = {}
    for r in records:
        label = r["case_label"]
        case_stats.setdefault(label, {"ok": 0, "total": 0})
        case_stats[label]["total"] += 1
        case_stats[label]["ok"] += int(r["status"] == "OK")

    print("\n" + "=" * 80)
    print(f"{'CASE LABEL':<55}  {'OK':>7}  {'RATE':>7}")
    print("-" * 80)
    for label in sorted(case_stats):
        d = case_stats[label]
        rate = (d["ok"] / d["total"] * 100) if d["total"] else 0
        print(f"{label:<55}  {d['ok']:>3}/{d['total']:<3}  {rate:>6.1f}%")
    print("=" * 80)


def _pct(n, d):
    return 0.0 if d <= 0 else (100.0 * n / d)


def summarize_status(records):
    total = len(records)
    ok_statuses = {"OK", "OK_AFTER_JSON_REPAIR"}
    ok = sum(1 for r in records if r.get("status") in ok_statuses)
    not_ok = total - ok
    return {
        "total": total,
        "ok": ok,
        "not_ok": not_ok,
        "ok_pct": round(_pct(ok, total), 3),
        "not_ok_pct": round(_pct(not_ok, total), 3),
    }


def build_status_report(all_results, active_cases):
    case_labels = [case_label for _, case_label, _ in active_cases]

    by_run_rows = {}
    by_run_case_rows = {}
    by_case_all_runs_rows = {label: [] for label in case_labels}

    for rec in all_results:
        run_name = rec.get("run_folder", "UNKNOWN_RUN")
        case_label = rec.get("case_label", "UNKNOWN_CASE")
        by_run_rows.setdefault(run_name, []).append(rec)
        by_run_case_rows.setdefault(run_name, {}).setdefault(case_label, []).append(rec)
        by_case_all_runs_rows.setdefault(case_label, []).append(rec)

    by_run = {run: summarize_status(rows) for run, rows in sorted(by_run_rows.items())}

    by_run_case = {}
    for run in by_run:
        by_run_case[run] = {}
        for case_label in case_labels:
            rows = by_run_case_rows.get(run, {}).get(case_label, [])
            by_run_case[run][case_label] = summarize_status(rows)

    by_case_all_runs = {}
    for case_label in case_labels:
        by_case_all_runs[case_label] = summarize_status(by_case_all_runs_rows.get(case_label, []))

    return {
        "overall": summarize_status(all_results),
        "by_run": by_run,
        "by_run_case": by_run_case,
        "by_case_all_runs": by_case_all_runs,
    }


def write_report_json(data, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(data, out_path)


def build_detailed_report(all_results):
    rows = sorted(
        all_results,
        key=lambda r: (
            str(r.get("run_folder", "")),
            int(r.get("case_id", 0)),
            int(r.get("problem_index", 0)),
        ),
    )
    details = []
    for r in rows:
        model_output = r.get("model_output") or {}
        details.append(
            {
                "run_folder": r.get("run_folder", ""),
                "case_id": r.get("case_id", ""),
                "case_label": r.get("case_label", ""),
                "problem_index": r.get("problem_index", ""),
                "status": r.get("status", ""),
                "triples_count": len(r.get("triples", []) or []),
                "answer": str(model_output.get("ANSWER", "")).strip(),
            }
        )
    return details


def build_segregated_output(all_results, active_cases):
    """
    Group records as run -> case -> records so main output is segregated.
    """
    case_order = [case_label for _, case_label, _ in active_cases]
    case_id_lookup = {case_label: case_id for case_id, case_label, _ in active_cases}

    runs_map = {}
    for rec in all_results:
        run_name = rec.get("run_folder", "UNKNOWN_RUN")
        case_label = rec.get("case_label", "UNKNOWN_CASE")
        runs_map.setdefault(run_name, {}).setdefault(case_label, []).append(rec)

    runs = []
    for run_name in sorted(runs_map):
        run_cases = runs_map[run_name]
        case_blocks = []

        for case_label in case_order:
            records = sorted(
                run_cases.get(case_label, []),
                key=lambda r: int(r.get("problem_index", 0)),
            )
            case_blocks.append(
                {
                    "case_id": case_id_lookup.get(case_label),
                    "case_label": case_label,
                    "summary": summarize_status(records),
                    "records": records,
                }
            )

        run_records = [rec for block in case_blocks for rec in block["records"]]
        runs.append(
            {
                "run_folder": run_name,
                "summary": summarize_status(run_records),
                "cases": case_blocks,
            }
        )

    return {
        "overall": summarize_status(all_results),
        "runs": runs,
    }


def solve_all(selected_indices=None):
    validate_provider_config()
    selected_indices = set(selected_indices or [])
    active_cases = list(CASES)
    selected_case_labels = {normalize_case_name(case_label) for _, case_label, _ in active_cases}

    if not QUESTION_CSV.exists():
        raise FileNotFoundError(f"Question file not found: {QUESTION_CSV}")

    if MODEL_PROVIDER == "ollama":
        available_models = check_ollama_running()
        if available_models is None:
            raise SystemExit(
                "Cannot reach Ollama at http://localhost:11434. Make sure Ollama is running (`ollama serve`)."
            )

        if OLLAMA_MODEL not in available_models:
            print(f"Warning: model '{OLLAMA_MODEL}' not listed in local models: {available_models}")
            print(f"If needed, run: ollama pull {OLLAMA_MODEL}")

    questions = read_questions(QUESTION_CSV)
    search_dirs = [BASE_DIR]
    if RUN_FOLDERS:
        search_dirs = [BASE_DIR / folder for folder in RUN_FOLDERS]
        missing_dirs = [str(d) for d in search_dirs if not d.exists()]
        if missing_dirs:
            raise FileNotFoundError(
                "These RUN_FOLDERS do not exist under FINAL OUTPUT: " + ", ".join(missing_dirs)
            )

    if selected_indices:
        questions = {idx: q for idx, q in questions.items() if idx in selected_indices}

    if not questions:
        raise SystemExit("No matching questions found for the given --index filter.")
    run_dirs = search_dirs
    total_calls = len(run_dirs) * len(active_cases) * len(questions)
    print(
        f"Provider: {MODEL_PROVIDER} | Model: "
        f"{OPENROUTER_MODEL if MODEL_PROVIDER == 'openrouter' else OLLAMA_MODEL}"
    )
    print(
        f"Retry settings: max {MAX_RETRIES} attempts, {RETRY_DELAY}s initial delay, {RETRY_BACKOFF}x backoff."
    )
    print(
        f"Running {len(run_dirs)} runs x {len(active_cases)} cases x {len(questions)} indices = {total_calls} potential calls."
    )

    all_results = []
    done = 0
    for run_idx, run_dir in enumerate(run_dirs, start=1):
        print(f"\n{'=' * 90}")
        print(f"RUN {run_idx}/{len(run_dirs)}")
        print(f"Run folder: {run_dir}")
        print(f"{'=' * 90}")

        run_results = []
        for case_id, case_label, _ in active_cases:
            print(f"\n[Case {case_id:02d}] {case_label}")
            run_csv = run_dir / f"{case_label}.csv"
            if not run_csv.exists():
                print(f"  [SKIP] CSV not found: {run_csv.name}")
                continue

            by_index = read_triples_grouped_by_index(run_csv)
            case_results = []
            for idx in sorted(questions):
                question = questions.get(idx, "")
                if not question:
                    continue

                done += 1
                triples = by_index.get(idx, [])
                record = {
                    "run_folder": run_dir.name,
                    "source_csv": str(run_csv.relative_to(BASE_DIR)),
                    "case_id": case_id,
                    "case_label": case_label,
                    "problem_index": idx,
                    "question": question,
                    "triples": triples,
                }

                if not triples:
                    record["model_output"] = {"ANSWER": "", "REASONING": ""}
                    record["status"] = "NO_TRIPLES_FOR_INDEX"
                    print(
                        f"[Run {run_idx} | {done}/{total_calls}] Case {case_id:02d} | Index {idx:4d} | "
                        f"Status: {record['status']:<20} | Triples:  0"
                    )
                    case_results.append(record)
                    run_results.append(record)
                    all_results.append(record)
                    continue

                try:
                    raw = query_model(question, triples)
                    parsed = extract_json_from_text(raw)
                    if isinstance(parsed, dict):
                        answer = str(parsed.get("ANSWER", "")).strip()
                        reasoning = str(parsed.get("REASONING", "")).strip()
                        record["model_output"] = {"ANSWER": answer, "REASONING": reasoning}
                        record["status"] = "OK"
                    else:
                        # One recovery attempt: ask model to reformat prior output as strict JSON.
                        repaired_raw = query_json_fix_only(raw)
                        repaired_parsed = extract_json_from_text(repaired_raw)
                        if isinstance(repaired_parsed, dict):
                            answer = str(repaired_parsed.get("ANSWER", "")).strip()
                            reasoning = str(repaired_parsed.get("REASONING", "")).strip()
                            record["model_output"] = {"ANSWER": answer, "REASONING": reasoning}
                            record["status"] = "OK_AFTER_JSON_REPAIR"
                            record["raw_response"] = raw
                            record["json_repair_response"] = repaired_raw
                        else:
                            record["model_output"] = {"ANSWER": "", "REASONING": ""}
                            record["status"] = "INVALID_JSON"
                            record["raw_response"] = raw
                            record["json_repair_response"] = repaired_raw
                except Exception as exc:
                    record["model_output"] = {"ANSWER": "", "REASONING": ""}
                    record["status"] = "ERROR"
                    record["error"] = str(exc)

                print(
                    f"[Run {run_idx} | {done}/{total_calls}] Case {case_id:02d} | Index {idx:4d} | "
                    f"Status: {record['status']:<20} | Triples: {len(triples):>2}"
                )
                case_results.append(record)
                run_results.append(record)
                all_results.append(record)

            case_json = save_case_json(REPORT_DIR / run_dir.name, case_label, case_results)
            print(f"  Saved case JSON: {case_json}")

        print("\nRun summary:")
        if run_results:
            print_summary(run_results)
        else:
            print("No records generated for this run.")

    segregated_output = build_segregated_output(all_results, active_cases)
    save_json(segregated_output, OUTPUT_JSON)
    save_json(all_results, OUTPUT_JSON_FLAT)
    save_json(all_results, COMPARE_INPUT_JSON)
    print(f"\nSaved segregated JSON output: {OUTPUT_JSON}")
    print(f"Saved flat JSON output ({len(all_results)} records): {OUTPUT_JSON_FLAT}")
    print(f"Saved compare input JSON: {COMPARE_INPUT_JSON}")

    summary = build_status_report(all_results, active_cases)
    summary_json = REPORT_DIR / "summary.json"
    detailed_json = REPORT_DIR / "detailed.json"
    write_report_json(summary, summary_json)
    write_report_json(build_detailed_report(all_results), detailed_json)

    overall = summary["overall"]
    print("\n=== Overall ===")
    print(
        f"Total={overall['total']} | OK={overall['ok']} ({overall['ok_pct']}%) | "
        f"NOT_OK={overall['not_ok']} ({overall['not_ok_pct']}%)"
    )

    print("\n=== By run ===")
    for run, stats in summary.get("by_run", {}).items():
        print(
            f"{run}: Total={stats['total']} | OK={stats['ok']} ({stats['ok_pct']}%) | "
            f"NOT_OK={stats['not_ok']} ({stats['not_ok_pct']}%)"
        )

    print(
        f"\nWrote:\n"
        f"  {COMPARE_INPUT_JSON}\n"
        f"  {summary_json}\n"
        f"  {detailed_json}\n"
    )


if __name__ == "__main__":
    solve_all(selected_indices=TARGET_INDICES)
