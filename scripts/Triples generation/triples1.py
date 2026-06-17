import os
import re
import csv
import json
import base64
import time
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-4-scout"
# MODEL = "google/gemma-3-4b-it"
# MODEL = "anthropic/claude-sonnet-4-5"
# MODEL = "google/gemini-3-flash-preview"
# MODEL = "mistralai/mistral-small-3.2-24b-instruct"

# Retry settings
MAX_RETRIES = 5
RETRY_DELAY = 5
RETRY_BACKOFF = 2

# Input modality mode:
# - "image_only": send only image
# - "image_plus_text": send image + external question text
# - "auto": send question text only when available for that index
INPUT_MODE = "image_only"

# Optional question text source for image_plus_text / auto mode.
# CSV/JSON supported.
QUESTION_TEXT_FILE = "./questions.csv"

# Run output folders (one folder per full run)
RUN_OUTPUT_DIRS = [
    r"./FINAL OUTPUT/run-1-llama-remaining",
    # r"./FINAL OUTPUT/run-2-mistral",
    # r"./FINAL OUTPUT/run-3-llama-remaining",
]

# 20 cases
CASES = [
    # Baseline
    # (1, "full_text + full_diagram", "./Dataset/Setting_1&2/FTFD"),
    # (11, "half_text + half_diagram", "./Dataset/Setting_1&2/Half Text Half Diagram Images"),

    # Noisy text + full diagram (4 text noise types)
    # (2, "noisy_text[numeric] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Num Distracto_Q Full Image"),
    # (3, "noisy_text[punctuation] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Punctuation Swap_Q Full Image"),
    (4, "noisy_text[sentence] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Sent Distractor_Q Full Image"),
    # (5, "noisy_text[typo] + full_diagram", "./Dataset/Setting_3&4/Noisy_Full_Text_Full_Diagram/Typo_Q Full Image"),

    # Full text + noisy diagram (5 diagram noise types)
    # (6, "full_text + noisy_diagram[bg_change]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/background_noise_fullQ"),
    # (7, "full_text + noisy_diagram[blur]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/blur_noise_fullQ"),
    # (8, "full_text + noisy_diagram[illumination]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/illumination_noise_fullQ"),
    # (9, "full_text + noisy_diagram[irr_objects]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/irrelevant_objects_noise_fullQ"),
    # (10, "full_text + noisy_diagram[pixel_noise]", "./Dataset/Setting_3&4/Full_Text_Noisy_Full_Diagram/pixel_noise_fullQ"),

    # Noisy half text + half diagram (4 text noise types)
    # (12, "noisy_half_text[numeric] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Numeric_Distractor"),
    # (13, "noisy_half_text[punctuation] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Punctuation_Noise_Text"),
    # (14, "noisy_half_text[sentence] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Distractor_Noise_Text"),
    # (15, "noisy_half_text[typo] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Typo_Noise"),

    # Half text + noisy half diagram (5 diagram noise types)
    # (16, "half_text + noisy_half_diagram[bg_change]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/background_noise"),
    # (17, "half_text + noisy_half_diagram[blur]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/blur_noise"),
    # (18, "half_text + noisy_half_diagram[illumination]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/illumination_noise"),
    # (19, "half_text + noisy_half_diagram[irr_objects]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/irrelevant_object_noise"),
    # (20, "half_text + noisy_half_diagram[pixel_noise]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/pixel_noise"),
]

TARGET_INDICES = [
    # 1, 2, 3, 4, 7, 8, 9, 10, 11, 14,
    # 16, 17, 20, 22, 25, 26, 29, 30, 32,
    # 34, 36, 37, 38, 40, 43, 44, 45, 46, 48, 49, 50, 53, 55, 56,
    # 57, 58, 59, 60, 64, 72, 73, 78, 79, 81, 84, 86, 88, 95, 99, 108,
    # 109, 112, 114, 115, 117, 119, 120, 123, 128, 132, 133, 149, 150,
    # 151, 152, 153, 157, 165, 198, 207, 210, 211, 223, 239, 243,
    # 257, 264, 281, 327, 348, 364, 365,
    # 420, 432, 438, 441, 447, 449, 457, 
    438
]

SYSTEM_PROMPT = """
You are given a math problem (text and/or diagram).
Your task is to construct a Sequential Graph (SQG) by extracting information in the exact order you perceive it.
DO NOT solve, compute, derive, infer, calculate, or answer the question.
---
CORE PRINCIPLE (STREAMING)
Process the input like human perception:
As soon as you notice a new piece of information,
Immediately convert it into an atomic triple,
Output it,
Then continue.
Do NOT wait. Do NOT group information.

---
WHAT TO EXTRACT (ON-THE-FLY)
Whenever encountered, extract:
* Entities (A, B, C, O, AB, ∠AOB, etc.)
* Structural relations
* Quantitative attributes
* Explicit visual facts
* Query target
---
REPRESENTATION SCHEMA (MANDATORY)
1. Entities
* Use minimal symbolic names:
  * Points: A, B, C, O
  * Segments: AB
  * Angles: ∠AOB
* Do NOT wrap entities in functions
---
2. Relation Vocabulary (FIXED)
Use ONLY these relations:
* is_a
* lies_on
* intersects
* connects
* collinear_with
* center_of
* radius_of
* diameter_of
* parallel_to
* perpendicular_to
* tangent_to
* has_measure
* rotated_about
* mapped_to
---
3. Typing via Relations
Define entity types using:
* (A, is_a, point)
* (AB, is_a, segment)
* (∠AOB, is_a, angle)
---
4. Numeric Representation
* Use:
  * 30_deg
  * 5_units
* Avoid natural language
---
STRICT RULES
1. Immediate Output
   * Output triples as soon as perceived.
2. Triple Limit 
   * Maximum number of triples is 20. Stop extraction at 20. Do NOT exceed this under any circumstance.
3. Atomicity
   * Each triple = exactly ONE piece of information.
4. No Redundancy
   * Never repeat or restate a triple.
5. No Reasoning
   * Do not provide any conversational text, introductions, conclusions, or step-by-step reasoning.
6. Faithfulness
   * Only extract explicitly given or clearly visible information.
7. Represent unknown values using ?. The presence of ? is mandatory.
---
TRIPLE FORMAT
(subject, relation, object)
---
TERMINATION
Stop when no new information can be perceived.
---
FINAL_GRAPH:
(all triples in exact order of extraction)
"""


def encode_image(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_image(folder, idx):
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_question_texts(filepath):
    if not filepath or not os.path.exists(filepath):
        return {}

    texts = {}
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return texts

            index_key = "problem_index" if "problem_index" in reader.fieldnames else (
                "index" if "index" in reader.fieldnames else None
            )
            if index_key is None:
                return texts

            text_keys = ["question_text", "question", "text", "prompt"]
            text_key = next((k for k in text_keys if k in reader.fieldnames), None)
            if text_key is None:
                return texts

            for row in reader:
                try:
                    idx = int(str(row.get(index_key, "")).strip())
                except Exception:
                    continue
                txt = str(row.get(text_key, "")).strip()
                if txt:
                    texts[idx] = txt

    elif ext == ".json":
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                txt = str(v).strip()
                if txt:
                    texts[idx] = txt
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                raw_idx = item.get("problem_index", item.get("index"))
                try:
                    idx = int(str(raw_idx).strip())
                except Exception:
                    continue
                txt = str(
                    item.get("question_text", item.get("question", item.get("text", "")))
                ).strip()
                if txt:
                    texts[idx] = txt

    return texts


def build_user_text(question_text=None):
    use_text = False
    if INPUT_MODE == "image_plus_text":
        use_text = True
    elif INPUT_MODE == "auto" and question_text:
        use_text = True

    if use_text:
        return (
            "Extract all explicitly given facts from this math problem as ordered triples. "
            "Do NOT solve or compute anything — only transcribe what is stated. "
            f"Question text:\n{question_text or ''}"
        )
    else:
        return (
            "Extract all explicitly given facts from this math problem as ordered triples. "
            "Do NOT solve, compute, or derive anything — only output triples for information "
            "that is directly visible or stated in the input."
        )


def parse_triples(raw_text):
    """
    Parses model output to list of triples.
    Handles:
      - (subject, relation, object)   <- current prompt format
      - <subject, relation, object>
      - subject, relation, object     (bare line form)
      - slight whitespace variation
    """
    cleaned = (raw_text or "").strip()
    cleaned = cleaned.replace(r"\[", "").replace(r"\]", "")
    cleaned = cleaned.replace(r"\(", "").replace(r"\)", "")

    triples = []
    seen = set()

    # 1) Angle-bracket form: <subject, relation, object>
    pattern_angle = re.compile(r"<\s*([^,<>]+?)\s*,\s*([^,<>]+?)\s*,\s*([^<>]+?)\s*>")
    for m in pattern_angle.finditer(cleaned):
        subject = m.group(1).strip()
        predicate = m.group(2).strip()
        obj = m.group(3).strip()
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"subject": subject, "predicate": predicate, "object": obj})

    # 2) Parenthesized: (subject, relation, object)
    pattern_paren = re.compile(r"\(\s*([^,\(\)]+?)\s*,\s*([^,\(\)]+?)\s*,\s*([^\(\)]+?)\s*\)")
    for m in pattern_paren.finditer(cleaned):
        subject = m.group(1).strip()
        predicate = m.group(2).strip()
        obj = m.group(3).strip()
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"subject": subject, "predicate": predicate, "object": obj})

    # 3) Bare line form: subject, relation, object (no wrappers)
    for line in cleaned.splitlines():
        s = line.strip().strip("`").strip()
        if not s:
            continue
        m = re.match(r"^([^,<>()\[\]]+?)\s*,\s*([^,<>()\[\]]+?)\s*,\s*(.+?)$", s)
        if not m:
            continue
        subject = m.group(1).strip()
        predicate = m.group(2).strip()
        obj = m.group(3).strip()
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"subject": subject, "predicate": predicate, "object": obj})

    return triples


def query_model_with_retry(b64_image, question_text=None, attempt_label="", mime_type="image/jpeg"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mathverse-noise-injector.com",
        "X-Title": "MathVerse Triple Extractor",
    }

    user_text = build_user_text(question_text)

    payload = {
        "model": MODEL,
        "max_tokens": 4000,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ],
    }

    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"].strip()
                if content:
                    return content

            print(
                f"        [Attempt {attempt}/{MAX_RETRIES}] {attempt_label} "
                f"— empty response, retrying in {delay}s...\n        Response: {result}"
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            body = ""
            if e.response is not None:
                try:
                    body = e.response.json()
                except Exception:
                    body = e.response.text
            print(
                f"        [Attempt {attempt}/{MAX_RETRIES}] {attempt_label} "
                f"— HTTP {status}, retrying in {delay}s...\n        Response: {body}"
            )
        except Exception as e:
            print(
                f"        [Attempt {attempt}/{MAX_RETRIES}] {attempt_label} "
                f"— {type(e).__name__}: {e}, retrying in {delay}s..."
            )

        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= RETRY_BACKOFF

    print(f"        [GIVE UP] {attempt_label} — all {MAX_RETRIES} attempts failed.")
    return None


def extract_triples_for_image(idx, case_id, case_label, folder, question_text):
    image_path = find_image(folder, idx)
    if image_path is None:
        return {
            "problem_index": idx,
            "case_id": case_id,
            "case_label": case_label,
            "image_path": "",
            "status": "IMAGE_NOT_FOUND",
            "triples": [],
            "raw_response": "",
        }

    b64_image = encode_image(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    attempt_label = f"Case {case_id:02d} | Index {idx}"
    raw_response = query_model_with_retry(b64_image, question_text, attempt_label, mime_type)
    if raw_response is None:
        return {
            "problem_index": idx,
            "case_id": case_id,
            "case_label": case_label,
            "image_path": image_path,
            "status": "API_FAILED",
            "triples": [],
            "raw_response": "",
        }

    triples = parse_triples(raw_response)
    status = "OK" if triples else "NO_TRIPLES_PARSED"
    return {
        "problem_index": idx,
        "case_id": case_id,
        "case_label": case_label,
        "image_path": image_path,
        "status": status,
        "triples": triples,
        "raw_response": raw_response,
    }


def save_json(all_data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


def save_csv(all_data, filepath):
    fields = ["problem_index", "case_id", "case_label", "status", "subject", "predicate", "object"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in all_data:
            if entry["triples"]:
                for t in entry["triples"]:
                    writer.writerow(
                        {
                            "problem_index": entry["problem_index"],
                            "case_id": entry["case_id"],
                            "case_label": entry["case_label"],
                            "status": entry["status"],
                            "subject": t["subject"],
                            "predicate": t["predicate"],
                            "object": t["object"],
                        }
                    )
            else:
                writer.writerow(
                    {
                        "problem_index": entry["problem_index"],
                        "case_id": entry["case_id"],
                        "case_label": entry["case_label"],
                        "status": entry["status"],
                        "subject": "",
                        "predicate": "",
                        "object": "",
                    }
                )
            writer.writerow({field: "" for field in fields})


def sanitize_filename(name):
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "unnamed_case"


def save_case_outputs(case_id, case_label, case_data, run_output_dir):
    file_stem = sanitize_filename(case_label)
    json_path = os.path.join(run_output_dir, f"{file_stem}.json")
    csv_path = os.path.join(run_output_dir, f"{file_stem}.csv")
    save_json(case_data, json_path)
    save_csv(case_data, csv_path)
    return json_path, csv_path


def print_summary(all_data):
    case_stats = {}
    for r in all_data:
        cid = r["case_id"]
        case_stats.setdefault(cid, {"label": r["case_label"], "ok": 0, "total": 0})
        case_stats[cid]["total"] += 1
        case_stats[cid]["ok"] += int(r["status"] == "OK")

    print("\n" + "=" * 75)
    print(f"{'CASE':>4}  {'LABEL':<50}  {'OK':>7}  {'RATE':>7}")
    print("-" * 75)
    for cid in sorted(case_stats):
        d = case_stats[cid]
        rate = (d["ok"] / d["total"] * 100) if d["total"] else 0
        print(f"{cid:>4}  {d['label']:<50}  {d['ok']:>3}/{d['total']:<3}  {rate:>6.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not found in environment variables.")

    question_texts = load_question_texts(QUESTION_TEXT_FILE)

    total_calls = len(CASES) * len(TARGET_INDICES)
    print(
        f"Retry settings: max {MAX_RETRIES} attempts, {RETRY_DELAY}s initial delay, {RETRY_BACKOFF}x backoff.\n"
    )
    print(f"Input mode: {INPUT_MODE}")
    if INPUT_MODE in ("image_plus_text", "auto"):
        print(f"Loaded {len(question_texts)} question texts from {QUESTION_TEXT_FILE}.\n")

    print(
        f"Running {len(RUN_OUTPUT_DIRS)} runs x {len(CASES)} cases x {len(TARGET_INDICES)} indices "
        f"= {len(RUN_OUTPUT_DIRS) * total_calls} total extractions."
    )

    for run_idx, run_output_dir in enumerate(RUN_OUTPUT_DIRS, start=1):
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"\n{'=' * 90}")
        print(f"RUN {run_idx}/{len(RUN_OUTPUT_DIRS)}")
        print(f"Output folder: {run_output_dir}")
        print(f"{'=' * 90}")

        all_data = []
        done = 0

        for case_id, case_label, folder in CASES:
            case_data = []
            folder_exists = os.path.isdir(folder)
            if not folder_exists:
                print(f"\n[Case {case_id:02d}] {case_label}")
                print(f"  [SKIP] Folder not found: {folder}")

            for idx in sorted(TARGET_INDICES):
                done += 1

                if not folder_exists:
                    result = {
                        "problem_index": idx,
                        "case_id": case_id,
                        "case_label": case_label,
                        "image_path": "",
                        "status": "FOLDER_NOT_FOUND",
                        "triples": [],
                        "raw_response": "",
                    }
                    case_data.append(result)
                    all_data.append(result)
                    continue

                q_text = question_texts.get(idx, "")
                result = extract_triples_for_image(idx, case_id, case_label, folder, q_text)
                case_data.append(result)
                all_data.append(result)

                print(
                    f"[Run {run_idx} | {done}/{total_calls}] Case {case_id:02d} | Index {idx:4d} | "
                    f"Status: {result['status']:<16} | Triples: {len(result['triples']):>2}"
                )

            case_json, case_csv = save_case_outputs(case_id, case_label, case_data, run_output_dir)
            print(f"  Saved case outputs: {case_json} | {case_csv}")

        print("\nRun summary:")
        print_summary(all_data)
