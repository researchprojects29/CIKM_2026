import os
import re
import csv
import json
import base64
import time
import filetype
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL = "meta-llama/llama-4-scout"
# MODEL = "google/gemma-3-4b-it"
MODEL = "anthropic/claude-sonnet-4-5"
 
# Input modality mode:
# - "image_only": send only image
# - "image_plus_text": send image + external question text
# - "auto": send question text only when available for that index
INPUT_MODE = "image_only"
 
# Optional question text source for image_plus_text / auto mode.
# CSV/JSON supported.
QUESTION_TEXT_FILE = "./questions.csv"
 
# Output paths
OUTPUT_JSON = "./triples_cases_2-348.json"
OUTPUT_CSV  = "./triples_cases_2-348.csv"
 
# Cases 11..20 from evaluate.py
CASES = [
    # (1, "half_text + half_diagram", "./Dataset/Setting_1&2/Half Text Half Diagram Images"),
    # (2, "noisy_half_text[numeric] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Numeric_Distractor"),
    # (3, "noisy_half_text[punctuation] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Punctuation_Noise_Text"),
    # (4, "noisy_half_text[sentence] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Distractor_Noise_Text"),
    # (5, "noisy_half_text[typo] + half_diagram", "./Dataset/Setting_5&6/Noisy_half_text_Half_Diagram/Typo_Noise"),
    # (6, "half_text + noisy_half_diagram[bg_change]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/background_noise"),
    # (7, "half_text + noisy_half_diagram[blur]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/blur_noise"),
    # (8, "half_text + noisy_half_diagram[illumination]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/illumination_noise"),
    # (9, "half_text + noisy_half_diagram[irr_objects]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/irrelevant_object_noise"),
    # (10, "half_text + noisy_half_diagram[pixel_noise]", "./Dataset/Setting_5&6/Half_text_Noisy_half_Diagram/pixel_noise"),
    (11, "full_text + full_diagram", "./Dataset/Setting_1&2/FTFD"),
]
 
TARGET_INDICES = [
    1, 2, 3, 4, 7,8, 
    9, 10, 11, 16, 17, 20, 22, 25,
        26, 
        29, 30, 84, 
        112, 114, 
        117, 119, 133, 149, 
        198, 243,348
        ,207, 210, 211, 239,  432
#     14, 16, 17, 20, 22, 25, 26, 29, 30,
#     32, 34, 36, 37, 38, 40, 43, 44, 45, 46, 48, 49, 50, 53, 55, 56,
#     57, 58, 59, 60, 64, 72, 73, 78, 79, 81, 84, 86, 88, 95, 99, 108,
#     109, 112, 114, 115, 117, 119, 120, 123, 128, 132, 133, 149, 150,
#     151, 152, 153, 157, 165, 198, 207, 210, 211, 223, 239, 243,
#     257, 264, 281, 327, 348, 364, 365, 420, 432, 438, 441, 447, 449, 457, 472
]
 
SYSTEM_PROMPT = """
You are given a math problem (text and/or diagram).
 
Your task is to construct a Sequential Graph (SQG) by extracting information in the exact order you perceive it.
 
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
 
1. Immediate Emission
 
   * Output triples as soon as perceived
 
2. Atomicity
 
   * Each triple = exactly ONE fact
 
3. No Redundancy
 
   * Never repeat or restate a triple
 
4. Strict Order
 
   * Preserve perception order
 
5. No Early Reasoning
 
   * Do NOT solve or infer hidden properties
 
6. Faithfulness
 
   * Only extract explicitly given or clearly visible facts
 
7. Represent unknown values using ?. The presence of ? is mandatory.
 
---
 
 TRIPLE FORMAT
 
(subject, relation, object)
 
---
 
 TERMINATION
 
Stop when no new information can be perceived.
 
---
 
 OUTPUT FORMAT
(ABC, is_a, triangle) 
(A, is_a, point) 
(B, is_a, point) 
(C, is_a, point) 
(AB, is_a, segment) 
(AC, is_a, segment) 
(BC, is_a, segment) 
(AB, connects, A) 
(AB, connects, B) 
(AC, connects, A) 
(AC, connects, C) 
(BC, connects, B) 
(BC, connects, C) 
(∠A, is_a, angle) 
(∠A, has_measure, 36_deg) 
(AB, has_measure, AC) 
(D, is_a, point) 
(D, lies_on, AC) 
(E, is_a, point) 
(E, lies_on, AB) 
(perpendicular_bisector_AB, is_a, line) 
(perpendicular_bisector_AB, perpendicular_to, AB) 
(perpendicular_bisector_AB, intersects, AC) 
(perpendicular_bisector_AB, intersects, AB) 
(perpendicular_bisector_AB, intersects, BC) 
(D, lies_on, perpendicular_bisector_AB) 
(E, lies_on, perpendicular_bisector_AB) 
(perpendicular_bisector_AB, intersects, AC) 
(perpendicular_bisector_AB, intersects, AB) 
(∠BDC, is_a, angle) 
(∠BDC, has_measure, ?) 
 
 
 
FINAL_GRAPH:
(all triples in exact order of extraction)
"""
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def encode_image(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def find_image(folder, idx):
    """Checks for multiple extensions and returns path + detected extension."""
    # Added .png, .jpg, .jpeg, and uppercase variants
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for ext in extensions:
        path = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(path):
            return path, ext.lower()
    return None, None

def get_mime_type_robust(filepath):
    # """Detects the real MIME type based on file content using the filetype library."""
    kind = filetype.guess(filepath)
    
    # If filetype successfully identifies the file
    if kind is not None:
        return kind.mime
    
    # Fallback to extension if content detection fails
    ext = os.path.splitext(filepath)[1].lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    return mapping.get(ext, "application/octet-stream")

def build_user_content(b64_image, mime_type, question_text=None):
    """Correctly structures the payload with the specific MIME type."""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
        }
    ]

    use_text = (INPUT_MODE == "image_plus_text") or (INPUT_MODE == "auto" and question_text)

    if use_text:
        text_content = f"Solve the maths problem and extract all relevant facts as ordered triples. Question text:\n{question_text or ''}"
    else:
        text_content = "Solve the maths problem and return only ordered triples in the same order as the information to solve the question is perceived by you."
    
    content.append({"type": "text", "text": text_content})
    return content
 
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
 
 
# def build_user_content(b64_image, mime_type, question_text=None):
#     content = [
#         {
#             "type": "image_url",
#             "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
#         }
#     ]
 
#     use_text = False
#     if INPUT_MODE == "image_plus_text":
#         use_text = True
#     elif INPUT_MODE == "auto" and question_text:
#         use_text = True
 
#     if use_text:
#         content.append(
#             {
#                 "type": "text",
#                 "text": (
#                     "Solve the maths problem and extract all relevant facts as ordered triples. "
#                     f"Question text:\n{question_text or ''}"
#                 ),
#             }
#         )
#     else:
#         content.append(
#             {
#                 "type": "text",
#                 "text": "Solve the maths problem and return only ordered triples in the same order as the information to solve the question is perceived by you.",
#             }
#         )
 
#     return content
 
 
def parse_triples(raw_text):
    """
    Parses model output to list of triples.
    Handles:
      - (subject, relation, object)   ← current prompt format
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
        subject   = m.group(1).strip()
        predicate = m.group(2).strip()
        obj       = m.group(3).strip()
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"subject": subject, "predicate": predicate, "object": obj})
 
    # 2) Parenthesized: (subject, relation, object)
    pattern_paren = re.compile(r"\(\s*([^,\(\)]+?)\s*,\s*([^,\(\)]+?)\s*,\s*([^\(\)]+?)\s*\)")
    for m in pattern_paren.finditer(cleaned):
        subject   = m.group(1).strip()
        predicate = m.group(2).strip()
        obj       = m.group(3).strip()
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
        subject   = m.group(1).strip()
        predicate = m.group(2).strip()
        obj       = m.group(3).strip()
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        triples.append({"subject": subject, "predicate": predicate, "object": obj})
 
    return triples
 
 
# ── API call — single attempt, full error logging ─────────────────────────────
 
def query_model(b64_image, mime_type, question_text=None, attempt_label=""):
    """
    Single API call with no retries.
    Logs the full HTTP error body on any failure so the real cause is visible.
    Returns the raw response string, or None on any failure.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://mathverse-noise-injector.com",
        "X-Title":       "MathVerse Triple Extractor",
    }
 
    payload = {
        "model":      MODEL,
        "max_tokens": 4000,
        "messages": [
            {
                "role":    "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role":    "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                    # user text part (may be just the task instruction or include q-text)
                    *build_user_content(b64_image, mime_type, question_text)[1:],
                ],
            },
        ],
    }
 
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
 
        # Always log full body on non-2xx so the real error is visible
        if not response.ok:
            print(
                f"        [FAIL] {attempt_label}\n"
                f"               HTTP {response.status_code} {response.reason}\n"
                f"               Body: {response.text[:800]}"
            )
            return None
 
        result = response.json()
 
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"]["content"].strip()
            if content:
                return content
            print(f"        [WARN] {attempt_label} — model returned empty content.")
            return None
 
        print(f"        [WARN] {attempt_label} — no 'choices' in response: {result}")
        return None
 
    except requests.exceptions.Timeout:
        print(f"        [FAIL] {attempt_label} — request timed out (120s).")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"        [FAIL] {attempt_label} — connection error: {e}")
        return None
    except Exception as e:
        print(f"        [FAIL] {attempt_label} — {type(e).__name__}: {e}")
        return None
 
 
# ── Per-image extraction ──────────────────────────────────────────────────────
 
def extract_triples_for_image(idx, case_id , case_label, folder, question_text):
    image_path, ext = find_image(folder, idx)
 
    if image_path is None:
        return {
            "problem_index": idx,
            "case_id":       case_id,
            "case_label":    case_label,
            "image_path":    "",
            "status":        "IMAGE_NOT_FOUND",
            "triples":       [],
            "raw_response":  "",
        }
    mime_type= get_mime_type_robust(image_path)
    b64_image     = encode_image(image_path)
    attempt_label = f"Case {case_id:02d} | Index {idx}"
    raw_response  = query_model(b64_image, mime_type, question_text, attempt_label)
 
    if raw_response is None:
        return {
            "problem_index": idx,
            "case_id":       case_id,
            "case_label":    case_label,
            "image_path":    image_path,
            "status":        "API_FAILED",
            "triples":       [],
            "raw_response":  "",
        }
 
    triples = parse_triples(raw_response)
    status  = "OK" if triples else "NO_TRIPLES_PARSED"
 
    return {
        "problem_index": idx,
        "case_id":       case_id,
        "case_label":    case_label,
        "image_path":    image_path,
        "status":        status,
        "triples":       triples,
        "raw_response":  raw_response,
    }
 
 
# ── Output helpers ────────────────────────────────────────────────────────────
 
def save_json(all_data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

 
def save_csv(all_data, filepath):
    fields = [
        "problem_index", "case_id", "case_label",
        "status", "subject", "predicate", "object",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in all_data:
            if entry["triples"]:
                for t in entry["triples"]:
                    writer.writerow({
                        "problem_index": entry["problem_index"],
                        "case_id":       entry["case_id"],
                        "case_label":    entry["case_label"],
                        "status":        entry["status"],
                        "subject":       t["subject"],
                        "predicate":     t["predicate"],
                        "object":        t["object"],
                    })
            else:
                writer.writerow({
                    "problem_index": entry["problem_index"],
                    "case_id":       entry["case_id"],
                    "case_label":    entry["case_label"],
                    "status":        entry["status"],
                    "subject":       "",
                    "predicate":     "",
                    "object":        "",
                })
 
 
def print_summary(all_data):
    case_stats = {}
    for r in all_data:
        cid = r["case_id"]
        case_stats.setdefault(cid, {"label": r["case_label"], "ok": 0, "total": 0})
        case_stats[cid]["total"] += 1
        case_stats[cid]["ok"]    += int(r["status"] == "OK")
 
    print("\n" + "=" * 75)
    print(f"{'CASE':>4}  {'LABEL':<50}  {'OK':>7}  {'RATE':>7}")
    print("-" * 75)
    for cid in sorted(case_stats):
        d    = case_stats[cid]
        rate = (d["ok"] / d["total"] * 100) if d["total"] else 0
        print(f"{cid:>4}  {d['label']:<50}  {d['ok']:>3}/{d['total']:<3}  {rate:>6.1f}%")
    print("=" * 75)
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not found in environment variables.")
 
    question_texts = load_question_texts(QUESTION_TEXT_FILE)
 
    total_calls = len(CASES) * len(TARGET_INDICES)
    print(f"Model          : {MODEL}")
    print(f"Input mode     : {INPUT_MODE}")
    print(f"Total calls    : {total_calls}  ({len(CASES)} cases × {len(TARGET_INDICES)} indices)")
    if INPUT_MODE in ("image_plus_text", "auto"):
        print(f"Question texts : {len(question_texts)} loaded from {QUESTION_TEXT_FILE}")
    print()
 
    all_data = []
    done     = 0
 
    for case_id, case_label, folder in CASES:
        folder_exists = os.path.isdir(folder)
 
        print(f"\n{'─'*60}")
        print(f"Case {case_id:02d}: {case_label}")
        if not folder_exists:
            print(f"  [SKIP] Folder not found: {folder}")
 
        for idx in sorted(TARGET_INDICES):
            done += 1
 
            if not folder_exists:
                all_data.append({
                    "problem_index": idx,
                    "case_id":       case_id,
                    "case_label":    case_label,
                    "image_path":    "",
                    "status":        "FOLDER_NOT_FOUND",
                    "triples":       [],
                    "raw_response":  "",
                })
                continue
 
            q_text = question_texts.get(idx, "")
            result = extract_triples_for_image(idx, case_id, case_label, folder, q_text)
            all_data.append(result)
 
            print(
                f"  [{done:4d}/{total_calls}]  "
                f"Q{idx:<5}  "
                f"Status: {result['status']:<18}  "
                f"Triples: {len(result['triples']):>3}"
            )
 
            # Incremental save after every call — no progress lost on crash
            save_json(all_data, OUTPUT_JSON)
            save_csv(all_data,  OUTPUT_CSV)
 
    print(f"\nSaved → {OUTPUT_JSON}")
    print(f"Saved → {OUTPUT_CSV}")
    print_summary(all_data)
 