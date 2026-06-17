# """
# evaluate_triple_sufficiency_no_answers.py
# ==========================================
# Judges whether model-generated Knowledge Triples are logically sufficient to
# derive an answer for each ARS (Atomic Reasoning Step) perception question,
# using Llama 4 Scout as the judge via OpenRouter.

# Unlike the original script, this variant:
#   - Does NOT require gold answers in the ARS JSON
#   - Does NOT compute accuracy / Ct scores
#   - Outputs only: problem_index, ars_id, question, answer, reasoning,
#                   triples_used, triples_count, unique_triples_count,
#                   which_triples_used

# ─────────────────────────────────────────────────────────────────────────────
# QUICK-START
# ─────────────────────────────────────────────────────────────────────────────
#   pip install openai pandas numpy python-dotenv tqdm
#   export OPENROUTER_API_KEY="YOUR_API_KEY"
#   python evaluate_triple_sufficiency_no_answers.py
# ─────────────────────────────────────────────────────────────────────────────
# """

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    sys.exit(
        "[ERROR] Install the openai package:  pip install openai\n"
        "        (OpenRouter uses the OpenAI-compatible API.)"
    )

try:
    from tqdm.asyncio import tqdm as atqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG  –  Edit everything here; do NOT change anything below this block
# ═════════════════════════════════════════════════════════════════════════════

# ── API ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = "YOUR_API_KEY"  # or paste key here
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "mistralai/mistral-small-3.2-24b-instruct"
# "mistralai/mistral-small-3.2-24b-instruct"
# "meta-llama/llama-4-scout"
# "google/gemma-3-4b-it"
# ── File paths ────────────────────────────────────────────────────────────
SCRIPT_DIR           = Path(__file__).resolve().parent
WORKSPACE            = SCRIPT_DIR.parent
ARS_JSON             = WORKSPACE / "ground_truth_triples" / "aps_perceive_questions_trial_test_questions_only.json"
TRIPLES_SOURCE_BASE  = WORKSPACE / "Triples" / "Split_Dataset"
OUTPUT_BASE          = WORKSPACE / "Triples_APS_evaluation_Split_Dataset"

# Which model folders to evaluate (each has its own run subfolders)
ACTIVE_MODELS: List[str] = ["Mistral - Split"]  # empty → use all models from MODEL_RUNS below

# All run folders per model (reference list)
MODEL_RUNS: Dict[str, List[str]] = {
    "Mistral - Split": ["run-1-mistral", "run-2-mistral", "run-3-mistral"],
}

# Which runs to process NOW (empty dict → use all runs from MODEL_RUNS above)
ACTIVE_RUNS: Dict[str, List[str]] = {
    "Mistral - Split": ["run-3-mistral"]
}

# Resume: skip a setting when BOTH result files already exist
SKIP_EXISTING_RESULTS = True

# Optional: process only these setting tags (empty → all settings in the run)
ONLY_SETTING_TAGS: List[str] = []

# Maps setting CSV stem → result-file tag (e.g. results_FTFD_APS.csv)
SETTING_RESULT_TAGS: Dict[str, str] = {
    "full_text + full_diagram":                      "FTFD",
    "half_text + half_diagram":                      "HTHD",
    "noisy_text[numeric] + full_diagram":            "NTNFD",
    "noisy_text[punctuation] + full_diagram":        "NTPFD",
    "noisy_text[sentence] + full_diagram":           "NTSFD",
    "noisy_text[typo] + full_diagram":               "NTTFD",
    "full_text + noisy_diagram[bg_change]":          "FTNDBGC",
    "full_text + noisy_diagram[blur]":               "FTNDBLUR",
    "full_text + noisy_diagram[illumination]":       "FTNDILL",
    "full_text + noisy_diagram[irr_objects]":        "FTNDIRR",
    "full_text + noisy_diagram[pixel_noise]":        "FTNDPN",
    "noisy_half_text[numeric] + half_diagram":       "NHTNHD",
    "noisy_half_text[punctuation] + half_diagram":   "NHTPHD",
    "noisy_half_text[sentence] + half_diagram":      "NHTSHD",
    "noisy_half_text[typo] + half_diagram":          "NHTTHD",
    "half_text + noisy_half_diagram[bg_change]":     "HTNHBGC",
    "half_text + noisy_half_diagram[blur]":          "HTNHBLUR",
    "half_text + noisy_half_diagram[illumination]":  "HTNHILL",
    "half_text + noisy_half_diagram[irr_objects]":   "HTNHIRR",
    "half_text + noisy_half_diagram[pixel_noise]":   "HTNHPN",
}

# ── Which problem indices to evaluate ────────────────────────────────────
# Leave empty → run ALL indices found in both files
PROBLEM_INDICES: List[int] = [
    1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 16, 17, 20, 22, 25, 26, 29, 30, 32, 34,
    36, 37, 38, 40, 43, 44, 45, 46, 48, 49, 50, 53, 55, 56, 57, 58, 59, 60,
    64, 72, 73, 78, 79, 81, 84, 86, 88, 95, 99, 108, 109, 112, 114, 115,
    117, 119, 120, 123, 128, 132, 133, 149, 150, 151, 152, 153, 157, 165,
    198, 207, 210, 211, 223, 239, 243, 257, 264, 281, 327, 348, 364, 365,
    420, 432, 438, 441, 447, 449, 457, 472
]

# ── Concurrency / retry ───────────────────────────────────────────────────
CONCURRENCY   = 10
MAX_RETRIES   = 10
BASE_BACKOFF  = 2.0   # seconds (doubles each retry, capped at MAX_BACKOFF)
MAX_BACKOFF   = 60.0

# ── Parse-error rectification ─────────────────────────────────────────────
JUDGE_MAX_TOKENS      = 8192
RETRY_ON_PARSE_ERROR  = True
PARSE_RETRY_BACKOFF   = 1.0
SHOW_PARSE_RETRY_LOGS = True

# If True, existing result files are opened, only problems with ParseError are
# re-judged, and the fixed entries are merged back into the same detail JSON.
PARSE_ERROR_ONLY_RERUN = True

# Terminal logging
TERMINAL_LOG_FILENAME = "Triples_Eval_APS.txt"
SAVE_TERMINAL_OUTPUT  = True
APPEND_TERMINAL_LOG   = True

# ═════════════════════════════════════════════════════════════════════════════


def _sanitize_setting_tag(case_label: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", case_label).strip("_").upper()
    return tag or "UNKNOWN"


def result_tag_for_case_label(case_label: str) -> str:
    return SETTING_RESULT_TAGS.get(case_label, _sanitize_setting_tag(case_label))


def source_run_dir(model_name: str, run_name: str) -> Path:
    return TRIPLES_SOURCE_BASE / model_name / run_name


def results_run_dir(model_name: str, run_name: str) -> Path:
    return OUTPUT_BASE / model_name / run_name


def terminal_log_path(model_name: str, run_name: str) -> Path:
    return results_run_dir(model_name, run_name) / TERMINAL_LOG_FILENAME


class _TeeStream:
    def __init__(self, stream: Any, log_file: Any) -> None:
        self.stream   = stream
        self.log_file = log_file

    def write(self, data: str) -> None:
        self.stream.write(data)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self) -> None:
        self.stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()


@contextmanager
def tee_terminal_log(log_path: Path) -> Iterator[None]:
    if not SAVE_TERMINAL_OUTPUT:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    append = APPEND_TERMINAL_LOG and log_path.exists()
    mode   = "a" if append else "w"

    with log_path.open(mode, encoding="utf-8") as log_f:
        if append:
            log_f.write("\n\n" + "=" * 70 + "\n")
            log_f.write(f"  SESSION RESUMED  —  {datetime.now().isoformat(timespec='seconds')}\n")
            log_f.write(f"  Judge model      : {MODEL}\n")
            log_f.write("=" * 70 + "\n\n")
        else:
            log_f.write("Triples APS Evaluation Log\n")
            log_f.write(f"Started     : {datetime.now().isoformat(timespec='seconds')}\n")
            log_f.write(f"Judge model : {MODEL}\n\n")

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, log_f)
        sys.stderr = _TeeStream(old_stderr, log_f)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def output_paths_for_setting(
    model_name: str,
    run_name: str,
    case_label: str,
) -> Tuple[Path, Path]:
    run_dir = results_run_dir(model_name, run_name)
    tag = result_tag_for_case_label(case_label)
    return (
        run_dir / f"results_{tag}_APS.csv",
        run_dir / f"results_{tag}_APS_detail.json",
    )


def discover_setting_csvs(run_dir: Path) -> List[Path]:
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("*.csv"), key=lambda p: p.name.lower())


def runs_to_process(model_name: str) -> List[str]:
    if model_name in ACTIVE_RUNS and ACTIVE_RUNS[model_name]:
        return ACTIVE_RUNS[model_name]
    return MODEL_RUNS.get(model_name, [])


def should_skip_setting(case_label: str, out_csv: Path, out_json: Path) -> bool:
    tag = result_tag_for_case_label(case_label)
    if ONLY_SETTING_TAGS and tag not in ONLY_SETTING_TAGS:
        return True
    if SKIP_EXISTING_RESULTS and out_csv.exists() and out_json.exists():
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Triple:
    idx: int
    subject: str
    predicate: str
    obj: str

    def as_text(self) -> str:
        return f"[T{self.idx}] ({self.subject}, {self.predicate}, {self.obj})"


@dataclass
class Problem:
    index: int
    triples: List[Triple]
    questions: Dict[str, str]          # qid → question text (no gold answers)

    # filled in after judging
    judgments: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  LOADING
# ─────────────────────────────────────────────────────────────────────────────

_CSV_KNOWN_COLS = frozenset({
    "problem_index", "case_id", "case_label", "status",
    "subject", "predicate", "object",
})
_PARALLEL_PREDICATES = frozenset({
    "parallel_to", "parallel", "is_parallel", "||",
})


def _repair_split_coordinate(row: dict) -> str:
    obj = (row.get("object") or "").strip()
    if not obj.startswith("(") or obj.endswith(")"):
        return obj

    extras: List[str] = []
    overflow = row.get(None)
    if overflow is not None:
        if isinstance(overflow, list):
            extras.extend(str(v).strip() for v in overflow if str(v).strip())
        else:
            extras.append(str(overflow).strip())

    for key, val in row.items():
        if key in _CSV_KNOWN_COLS or key is None or not val:
            continue
        extras.append(str(val).strip())

    if extras:
        return f"{obj},{extras[0]}"
    return obj


def _canonical_segment(name: str) -> Tuple[str, ...]:
    name = name.strip()
    if len(name) == 2 and name.isalpha():
        return tuple(sorted(name))
    return (name,)


def _parallel_pair_key(side_a: str, side_b: str) -> frozenset:
    return frozenset({_canonical_segment(side_a), _canonical_segment(side_b)})


def _has_parallel_triple(triples: List[Triple], side_a: str, side_b: str) -> bool:
    target = _parallel_pair_key(side_a, side_b)
    for triple in triples:
        if triple.predicate not in _PARALLEL_PREDICATES:
            continue
        if _parallel_pair_key(triple.subject, triple.obj) == target:
            return True
    return False


def _infer_parallelogram_parallel_triples(triples: List[Triple]) -> List[Triple]:
    inferred: List[Triple] = []
    next_idx = len(triples) + 1

    for triple in triples:
        if triple.predicate != "is_a" or triple.obj != "parallelogram":
            continue
        name = triple.subject
        if len(name) != 4 or not name.isalpha() or not name.isupper():
            continue

        a, b, c, d = name
        opposite_pairs = ((f"{a}{b}", f"{c}{d}"), (f"{b}{c}", f"{d}{a}"))

        for side_a, side_b in opposite_pairs:
            if _has_parallel_triple(triples + inferred, side_a, side_b):
                continue
            inferred.append(Triple(
                idx=next_idx,
                subject=side_a,
                predicate="parallel_to",
                obj=side_b,
            ))
            next_idx += 1

    return inferred


def _expand_inferred_triples(triples: List[Triple]) -> List[Triple]:
    inferred = _infer_parallelogram_parallel_triples(triples)
    if not inferred:
        return triples
    return triples + inferred


def load_triples(csv_path: Path) -> Dict[int, List[Triple]]:
    grouped: Dict[int, List[Triple]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("problem_index") or "").strip()
            if not raw:
                continue
            try:
                pidx = int(raw)
            except ValueError:
                continue

            subj = (row.get("subject")   or "").strip()
            pred = (row.get("predicate") or "").strip()
            obj  = _repair_split_coordinate(row)

            if not (subj or pred or obj):
                continue

            grouped[pidx].append(
                Triple(
                    idx=len(grouped[pidx]) + 1,
                    subject=subj,
                    predicate=pred,
                    obj=obj,
                )
            )

    return {pidx: _expand_inferred_triples(triples) for pidx, triples in grouped.items()}


def load_ars(json_path: Path) -> Dict[int, dict]:
    """
    Parse the ARS JSON. Only questions are needed — gold answers are ignored.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items() if str(k).strip().isdigit()}


def build_problems(
    triples_map: Dict[int, List[Triple]],
    ars_map: Dict[int, dict],
    selected: List[int],
) -> List[Problem]:
    available = sorted(set(triples_map) & set(ars_map))
    if selected:
        requested = set(selected)
        missing_triples = requested - set(triples_map)
        missing_ars     = requested - set(ars_map)
        if missing_triples:
            print(f"[warn] {len(missing_triples)} requested indices not in triples CSV: "
                  f"{sorted(missing_triples)}")
        if missing_ars:
            print(f"[warn] {len(missing_ars)} requested indices not in ARS JSON: "
                  f"{sorted(missing_ars)}")
        run_indices = sorted(requested & set(available))
    else:
        run_indices = available
        print(f"[info] No PROBLEM_INDICES specified – running all {len(run_indices)} "
              "available indices")

    problems: List[Problem] = []
    for pidx in run_indices:
        pdata = ars_map[pidx]
        
        # EXTRACT QUESTIONS DIRECTLY from the mapped dictionary
        questions = {
            qid: str(q_text).strip()
            for qid, q_text in pdata.items()
        }

        problems.append(Problem(
            index=pidx,
            triples=triples_map[pidx],
            questions=questions,
        ))
    return problems


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT CONSTRUCTION
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You will receive:
1) Perception Triples — these describe the geometry problem.
2) A JSON object of sub-questions labeled Q1, Q2, Q3, etc.

Use only the perception triples to answer each sub-question.

Return STRICT JSON only. The output must use the same Q labels as the input.
For each Q label, return an object with exactly these keys:
{
  "ANSWER": "...",
  "REASONING": "...",
  "triples_used": [<int>, ...],
  "triples_count": <int>
}

Rules:
1. Include only the triple indices actually used to derive the answer.
   Example: [1, 4, 7]

2. Do NOT use triples whose object is "?".
   These triples are placeholders for the final answer to be calculated, do not use these triples to answer APS questions.

3. triples_count must exactly equal the number of unique triples_used.

4. If the answer cannot be derived from the triples, set:
   "ANSWER": null

- Do not output markdown.
- Do not output extra keys.
- ANSWER must be concise.
- REASONING should briefly justify the answer from the perception triple.
- If a question cannot be answered from the perception triple, set ANSWER to null and explain why in REASONING.
"""

def _qsort(qid: str) -> Tuple[int, str]:
    m = re.match(r"Q(\d+)", qid)
    return (int(m.group(1)) if m else 9999, qid)


def build_user_prompt(p: Problem) -> str:
    triple_block = "\n".join(t.as_text() for t in p.triples) or "(no triples)"
    
    # Send the questions as a JSON object as instructed in System Prompt
    questions_json = json.dumps(p.questions, indent=2)

    return (
        "1) Perception Triples:\n"
        f"{triple_block}\n\n"
        "2) Sub-questions:\n"
        f"{questions_json}\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  RESPONSE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _find_json_object(text: str) -> Optional[str]:
    s = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    s = re.sub(r"\n?```$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            escape = (ch == "\\" and not escape)
            if ch == '"' and not escape:
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _escape_control_chars_inside_strings(text: str) -> str:
    out: List[str] = []
    in_str = False
    escape = False

    for ch in text:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_str = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 32:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_str = True

    return "".join(out)


def _coerce_missing_outer_braces(text: str) -> str:
    s = _strip_json_fences(text)
    if s.startswith("{"):
        return s
    
    # Heuristically look for Q labels if outer braces are missing
    key_pos = s.find('"Q1"')
    if key_pos < 0:
        key_pos = s.find('Q1"')
    if key_pos < 0:
        return s

    s = s[key_pos:].strip()
    if not s.startswith('"'):
        s = '"' + s
    s = "{" + s
    if not s.rstrip().endswith("}"):
        s = s.rstrip().rstrip(",") + "}"
    return s


def _candidate_json_blocks(raw: str) -> List[str]:
    candidates: List[str] = []
    stripped = _strip_json_fences(raw)

    for candidate in (
        _find_json_object(stripped),
        _coerce_missing_outer_braces(stripped),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    repaired: List[str] = []
    for candidate in candidates:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        cleaned = _escape_control_chars_inside_strings(cleaned)
        repaired.append(cleaned)

    return repaired


def _parse_triple_index(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def parse_response(raw: str, problem_index: int, valid_idx: Set[int], questions: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Parse the judge JSON response into a clean list of per-question dicts.
    The response format is now expected to be:
      {
         "Q1": { "ANSWER": "...", "REASONING": "...", "triples_used": [...], "triples_count": X },
         "Q2": ...
      }
    """
    obj: Optional[Dict[str, Any]] = None
    errors: List[str] = []

    for block in _candidate_json_blocks(raw):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as e:
            errors.append(str(e))
            continue

        if isinstance(parsed, dict):
            obj = parsed
            break
        errors.append("parsed JSON was not an object")

    if obj is None:
        if errors:
            raise ValueError("; ".join(errors[-2:]))
        raise ValueError("No JSON object found in model response")

    cleaned: List[Dict[str, Any]] = []
    for q_label, item in obj.items():
        if not isinstance(item, dict):
            continue
            
        ars_id = str(q_label).strip()
        if not ars_id:
            continue

        # Deduplicate + validate triple indices
        raw_triples = item.get("triples_used") or []
        if not isinstance(raw_triples, (list, tuple, set)):
            raw_triples = [raw_triples]

        triples_used: List[int] = []
        for t in raw_triples:
            ti = _parse_triple_index(t)
            if ti is None:
                continue
            if ti in valid_idx and ti not in triples_used:
                triples_used.append(ti)

        unique_count = len(set(triples_used))

        # Human-readable triple labels e.g. "T1, T3, T5"
        which_triples = ", ".join(f"T{i}" for i in sorted(triples_used)) if triples_used else ""

        cleaned.append({
            "problem_index":        problem_index,
            "ars_id":               ars_id,
            "question":             questions.get(ars_id, ""),
            "answer":               item.get("ANSWER"),
            "reasoning":            item.get("REASONING"),
            "triples_used":         triples_used,
            "triples_count":        len(triples_used),
            "unique_triples_count": unique_count,
            "which_triples_used":   which_triples,
        })
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
#  API CALL (with retry / back-off)
# ─────────────────────────────────────────────────────────────────────────────

async def judge_problem(
    client: AsyncOpenAI,
    problem: Problem,
    semaphore: asyncio.Semaphore,
) -> Problem:
    user_prompt = build_user_prompt(problem)
    valid_idx   = {t.idx for t in problem.triples}

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                raw = (resp.choices[0].message.content or "").strip()
                problem.raw_response = raw
                problem.judgments    = parse_response(raw, problem.index, valid_idx, problem.questions)
                return problem

            except (openai.RateLimitError, openai.APIStatusError) as e:
                status = getattr(e, "status_code", None)
                retriable = isinstance(e, openai.RateLimitError) or status in (
                    408, 429, 500, 502, 503, 504
                )
                if retriable and attempt < MAX_RETRIES:
                    delay = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1)))
                    delay += random.uniform(0, 1.5)
                    print(f"  [p{problem.index}] Retry {attempt}/{MAX_RETRIES} "
                          f"after {delay:.1f}s (status={status})", flush=True)
                    await asyncio.sleep(delay)
                    continue
                problem.error = f"{type(e).__name__}(status={status}): {e}"
                return problem

            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt < MAX_RETRIES:
                    delay = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1)))
                    print(f"  [p{problem.index}] Network error retry {attempt}: {e}",
                          flush=True)
                    await asyncio.sleep(delay)
                    continue
                problem.error = f"{type(e).__name__}: {e}"
                return problem

            except ValueError as e:
                if RETRY_ON_PARSE_ERROR and attempt < MAX_RETRIES:
                    delay = min(MAX_BACKOFF, PARSE_RETRY_BACKOFF * (2 ** (attempt - 1)))
                    delay += random.uniform(0, 0.75)
                    if SHOW_PARSE_RETRY_LOGS:
                        print(
                            f"  [p{problem.index}] Parse retry {attempt}/{MAX_RETRIES} "
                            f"after {delay:.1f}s: {e}",
                            flush=True,
                        )
                    await asyncio.sleep(delay)
                    continue
                problem.error = f"ParseError: {e}"
                return problem

            except Exception as e:
                problem.error = f"{type(e).__name__}: {e}"
                return problem

    problem.error = problem.error or "Exhausted all retries"
    return problem


async def judge_all(problems: List[Problem]) -> List[Problem]:
    if not OPENROUTER_API_KEY:
        sys.exit(
            "[ERROR] No API key found.\n"
            "  Set OPENROUTER_API_KEY as an env variable, or paste it into\n"
            "  the CONFIG block at the top of this file."
        )

    client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "[https://github.com/research/geometry-triples](https://github.com/research/geometry-triples)",
            "X-Title":      "Model Triple APS Sufficiency Judge",
        },
        timeout=120.0,
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [judge_problem(client, p, semaphore) for p in problems]

    try:
        if HAS_TQDM:
            results = await atqdm.gather(*tasks, total=len(tasks),
                                         desc="Judging problems")
        else:
            results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY (no accuracy / Ct — just counts)
# ─────────────────────────────────────────────────────────────────────────────

def has_non_null_answer(answer: Any) -> bool:
    """Return True only when the model provided a real answer."""
    if answer is None:
        return False
    text = str(answer).strip().lower()
    return text not in {
        "",
        "null",
        "none",
        "nan",
        "n/a",
        "na",
        "unknown",
        "not provided",
        "not specified",
    }


def print_summary(problems: List[Problem]) -> None:
    """Print a lightweight per-problem summary (no accuracy metrics)."""
    print()
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    total_q   = 0
    answered  = 0
    for p in problems:
        n_q  = len(p.questions)
        n_ans = sum(
            1 for j in (p.judgments or [])
            if has_non_null_answer(j.get("answer"))
        )
        total_q  += n_q
        answered += n_ans
        err = f"  ERROR: {p.error}" if p.error else ""
        print(f"  Problem {p.index:>4} | questions: {n_q} | answered: {n_ans}{err}")
    # print(problems)
    print()
    print(f"  Total questions : {total_q}")
    print(f"  Answered        : {answered}")
    print(f"  Unanswerable    : {total_q - answered}")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

# Flat CSV columns — exactly the requested fields
_CSV_COLUMNS = [
    "problem_index",
    "ars_id",
    "question",
    "answer",
    "reasoning",
    "triples_used",
    "triples_count",
    "unique_triples_count",
    "which_triples_used",
]


def save_outputs(
    problems: List[Problem],
    output_csv: Path,
    output_detail_json: Path,
) -> None:
    """
    Write two files:
      • A flat CSV  — one row per ARS question, columns = _CSV_COLUMNS
      • A detail JSON — one entry per problem with raw_response + judgments
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Flat CSV ─────────────────────────────────────────────────────────
    flat_rows: List[Dict[str, Any]] = []
    for p in problems:
        if p.error and not p.judgments:
            # Record a placeholder row so the problem is visible in the CSV
            for qid, question in sorted(p.questions.items(), key=lambda kv: _qsort(kv[0])):
                flat_rows.append({
                    "problem_index":        p.index,
                    "ars_id":               qid,
                    "question":             question,
                    "answer":               None,
                    "reasoning":            None,
                    "triples_used":         [],
                    "triples_count":        0,
                    "unique_triples_count": 0,
                    "which_triples_used":   "",
                })
            continue

        for j in sorted(p.judgments or [], key=lambda x: _qsort(x.get("ars_id", ""))):
            flat_rows.append({
                "problem_index":        j["problem_index"],
                "ars_id":               j["ars_id"],
                "question":             j["question"],
                "answer":               j["answer"],
                "reasoning":            j["reasoning"],
                "triples_used":         json.dumps(j["triples_used"]),   # JSON array as string
                "triples_count":        j["triples_count"],
                "unique_triples_count": j["unique_triples_count"],
                "which_triples_used":   j["which_triples_used"],
            })

    df = pd.DataFrame(flat_rows, columns=_CSV_COLUMNS)
    df.to_csv(output_csv, index=False)
    print(f"[saved] Flat CSV     → {output_csv}")

    # ── Detail JSON ───────────────────────────────────────────────────────
    output_detail_json.parent.mkdir(parents=True, exist_ok=True)
    detail = [
        {
            "problem_index": p.index,
            "error":         p.error,
            "raw_response":  p.raw_response,
            "judgments":     p.judgments,
        }
        for p in problems
    ]
    with output_detail_json.open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"[saved] Detail JSON  → {output_detail_json}")


# ─────────────────────────────────────────────────────────────────────────────
#  PARSE-ERROR REPAIR
# ─────────────────────────────────────────────────────────────────────────────

def is_parse_error_entry(entry: Dict[str, Any]) -> bool:
    error = entry.get("error")
    return isinstance(error, str) and error.startswith("ParseError:")


def detail_entry_to_problem(entry: Dict[str, Any], ars_map: Dict[int, dict]) -> Problem:
    pidx  = int(entry.get("problem_index"))
    pdata = ars_map.get(pidx, {}) or {}
    
    # Extract questions directly for repair process too
    questions = {
        qid: str(q_text).strip()
        for qid, q_text in pdata.items()
    }
    
    return Problem(
        index=pidx,
        triples=[],
        questions=questions,
        judgments=entry.get("judgments") or [],
        raw_response=entry.get("raw_response") or "",
        error=entry.get("error"),
    )


def problem_to_detail_entry(problem: Problem) -> Dict[str, Any]:
    return {
        "problem_index": problem.index,
        "error":         problem.error,
        "raw_response":  problem.raw_response,
        "judgments":     problem.judgments,
    }


def save_from_detail(
    detail: List[Dict[str, Any]],
    output_csv: Path,
    output_detail_json: Path,
    ars_map: Dict[int, dict],
) -> None:
    problems = [detail_entry_to_problem(e, ars_map) for e in detail]
    save_outputs(problems, output_csv, output_detail_json)


def repair_parse_errors_for_setting(
    triples_csv: Path,
    output_csv: Path,
    output_detail_json: Path,
    *,
    label: str = "",
) -> bool:
    if not output_detail_json.exists():
        print(f"[repair-skip] Detail JSON not found: {output_detail_json}")
        return False
    if not triples_csv.exists():
        print(f"[repair-skip] Triples CSV not found: {triples_csv}")
        return False

    with output_detail_json.open("r", encoding="utf-8") as f:
        detail = json.load(f)
    if not isinstance(detail, list):
        print(f"[repair-skip] Detail JSON is not a list: {output_detail_json}")
        return False

    failed_indices = sorted({
        int(entry["problem_index"])
        for entry in detail
        if is_parse_error_entry(entry) and str(entry.get("problem_index", "")).isdigit()
    })
    if not failed_indices:
        print(f"[repair-skip] {label or output_detail_json.name}: no ParseError entries")
        return False

    header = label or triples_csv.name
    print()
    print("=" * 70)
    print(f"  PARSE-ERROR REPAIR: {header}")
    print("=" * 70)
    print(f"[repair] Failed indices : {failed_indices}")

    triples_map = load_triples(triples_csv)
    ars_map     = load_ars(ARS_JSON)
    problems    = build_problems(triples_map, ars_map, failed_indices)

    if not problems:
        print("[repair-skip] No failed problems could be rebuilt for this CSV.")
        return False


    print('$$$$$$$$$$$$$$$$$$$$$$$')
    print('ABCD:',problems)
    print('$$$$$$$$$$$$$$$$$$$$$$$')    

    print(f"\n[info] {len(problems)} ParseError problem(s) queued  |  model = {MODEL}\n")

    repaired_results = asyncio.run(judge_all(problems))
    replacements = {p.index: problem_to_detail_entry(p) for p in repaired_results}

    merged_detail: List[Dict[str, Any]] = []
    replaced, still_failed = 0, 0
    for entry in detail:
        pidx = int(entry.get("problem_index"))
        if pidx in replacements:
            new_entry = replacements[pidx]
            merged_detail.append(new_entry)
            replaced += 1
            if is_parse_error_entry(new_entry):
                still_failed += 1
        else:
            merged_detail.append(entry)

    save_from_detail(merged_detail, output_csv, output_detail_json, ars_map)
    print(f"[repair] Replaced entries : {replaced}")
    print(f"[repair] Still ParseError : {still_failed}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE SETTING EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_single_evaluation(
    triples_csv: Path,
    output_csv: Path,
    output_detail_json: Path,
    *,
    label: str = "",
) -> None:
    if not triples_csv.exists():
        print(f"[skip] Triples CSV not found: {triples_csv}")
        return
    if not ARS_JSON.exists():
        sys.exit(f"[ERROR] ARS JSON not found: {ARS_JSON}")

    header = label or triples_csv.name
    print()
    print("=" * 70)
    print(f"  EVALUATING: {header}")
    print("=" * 70)
    print(f"[load] Triples CSV : {triples_csv}")
    print(f"[load] ARS JSON    : {ARS_JSON}")
    print(f"[out]  Results CSV : {output_csv}")
    print(f"[out]  Detail JSON : {output_detail_json}")

    triples_map = load_triples(triples_csv)
    ars_map     = load_ars(ARS_JSON)
    problems    = build_problems(triples_map, ars_map, PROBLEM_INDICES)

    if not problems:
        print("[warn] No problems to evaluate for this CSV – skipping.")
        return

    print(f"\n[info] {len(problems)} problems queued  |  model = {MODEL}  |  "
          f"concurrency = {CONCURRENCY}\n")

    results = asyncio.run(judge_all(problems))
    print_summary(results)
    save_outputs(results, output_csv, output_detail_json)


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_models_batch_evaluation() -> None:
    if not ARS_JSON.exists():
        sys.exit(f"[ERROR] ARS JSON not found: {ARS_JSON}")
    if not TRIPLES_SOURCE_BASE.exists():
        sys.exit(f"[ERROR] Triples source base not found: {TRIPLES_SOURCE_BASE}")

    print(f"[batch] Judge model     : {MODEL}")
    print(f"[batch] ARS JSON        : {ARS_JSON}")
    print(f"[batch] Triples source  : {TRIPLES_SOURCE_BASE}")
    print(f"[batch] Output base     : {OUTPUT_BASE}")
    print(f"[batch] Active models   : {', '.join(ACTIVE_MODELS)}")

    total_csvs = 0
    skipped    = 0

    for model_name in ACTIVE_MODELS:
        runs = runs_to_process(model_name)
        if not runs:
            print(f"[warn] No runs configured for model {model_name!r}")
            continue

        print(f"\n{'#' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'#' * 70}")

        for run_name in runs:
            log_path = terminal_log_path(model_name, run_name)
            with tee_terminal_log(log_path):
                print(f"[log] Terminal output → {log_path}")
                input_dir = source_run_dir(model_name, run_name)
                csv_files = discover_setting_csvs(input_dir)
                if not csv_files:
                    print(f"[warn] No CSV files in {input_dir}")
                    continue

                print(f"\n[batch] {model_name}/{run_name}: {len(csv_files)} setting CSV(s)")
                for csv_path in csv_files:
                    case_label = csv_path.stem
                    out_csv, out_json = output_paths_for_setting(
                        model_name, run_name, case_label
                    )

                    if PARSE_ERROR_ONLY_RERUN:
                        tag = result_tag_for_case_label(case_label)
                        if ONLY_SETTING_TAGS and tag not in ONLY_SETTING_TAGS:
                            print(f"[skip] {case_label} (not in ONLY_SETTING_TAGS)")
                            skipped += 1
                            continue
                        if out_json.exists():
                            repaired = repair_parse_errors_for_setting(
                                triples_csv=csv_path,
                                output_csv=out_csv,
                                output_detail_json=out_json,
                                label=f"{model_name}/{run_name} / {case_label}",
                            )
                            if repaired:
                                total_csvs += 1
                            else:
                                skipped += 1
                            continue
                        print(f"[repair] No existing detail for {case_label}; running full setting.")

                    if should_skip_setting(case_label, out_csv, out_json):
                        reason = (
                            "already exists"
                            if out_csv.exists() and out_json.exists()
                            else "not in ONLY_SETTING_TAGS"
                        )
                        print(f"[skip] {case_label} ({reason})")
                        skipped += 1
                        continue

                    run_single_evaluation(
                        triples_csv=csv_path,
                        output_csv=out_csv,
                        output_detail_json=out_json,
                        label=f"{model_name}/{run_name} / {case_label}",
                    )
                    total_csvs += 1

    print()
    print("=" * 70)
    print(f"  BATCH COMPLETE  —  {total_csvs} setting CSV(s) processed")
    print(f"  Skipped         : {skipped}")
    print(f"  Results root    : {OUTPUT_BASE}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_models_batch_evaluation()


if __name__ == "__main__":
    main()