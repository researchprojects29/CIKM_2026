# ars_generator_new.py
import os
import json
import base64
import time
import re
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4-5"

IMAGE_FOLDER = "Full Q Full Image"
OUTPUT_JSON = "ars_output_new_perceive.json"
REPORT_JSON = "ars_report.json"
TARGET_PROBLEMS = [1,2,3,4,7,8,9,10,11,16,17,20,22,25,26,27,29,30,112,117,119,133,149,198,207,210,211,239,243,348,432,
    34, 36, 37, 38,40,43,45,46,48,49,50,53,55,56,57,58,59,60,72,73,78,81,84,86,88,109,123,150,165,
    14, 32, 44, 64, 95, 99, 108, 114, 115, 120, 128, 132, 151, 152, 153, 157, 223, 257, 264, 281, 327, 364, 365, 438, 441, 447, 449, 457, 472,420
    ]

MAX_RETRIES = 5
RETRY_DELAY = 5
RETRY_BACKOFF = 2

SYSTEM_PROMPT = """
You are a Strategic Question Generator for complex visual reasoning problems.
Your goal is to generate a minimal and sufficient Auxiliary Reasoning Set (ARS) for a given
question. The ARS framework decomposes a complex reasoning problem into a sequence of structured
sub-questions. Each sub-question is directly answerable from the provided information, and together
they form a scaffold that enables a language model to solve the original question without ever accessing
the image. The sub-questions should provide any details that are necessary to answer the original
question. If there is a sub-question that is not answerable from the provided information, do not include
it in the ARS.
Core Principle
• Directly answerable using the image and original question.
• Concrete and visual, avoiding vague or abstract queries.
Structure of the ARS
S = {(q1, a1),(q2, a2), . . . ,(qn, an)}
Each sub-question must include:
• "question": A short, visually grounded sub-question.
• "depends_on_sub_question": List of sub-question IDs it depends on (e.g., [Q1, Q2]).
• "depends_on_image": yes/no.
• Most important: Must not reveal the final answer.
Required Properties:
1. Sufficiency: Enough to solve the original question.
2. Minimality: Removing any sub-question makes it unsolvable.
3. Dependency-restricted: Only reference allowed dependencies.
4. No redundancy or trivial sub-questions.
5. Answers must be concise: a numeric value or a single-word label. DO NOT answer with 'yes' or 'no'.
6. Do not duplicate the main question.

Additional rules for this generator (must be followed exactly):
A. Produce ONLY perception ("perception") sub-questions in the ARS. Do NOT include any
     reasoning/inferential sub-questions. Every sub-question object must include a field
     "type": "perception".
B. When validating/answering sub-questions, each answer must be a single token: a number or a
    single-word label (do NOT use 'yes' or 'no'). Labels should be lowercase alphabetical words with
    no spaces or punctuation (examples: parallel, perpendicular, collinear, intersecting, inside,
    outside, greater, equal). Do NOT provide explanations or multi-word answers.

Final Output Format (JSON only):
{
"Q1": {
"question": "...",
"depends_on_sub_question": [],
"depends_on_text": "Yes",
"depends_on_image": "No",
"type": "perception"
},
...
}

Respond ONLY with the JSON object (no surrounding markdown or commentary).
"""


def encode_image(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_image_mime(filepath):
    try:
        with open(filepath, "rb") as f:
            header = f.read(16)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
    except Exception:
        pass
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


def find_image_by_index(folder, idx):
    # Try exact numeric filenames first (e.g., "34.jpg")
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(path):
            return path

    # Fallback: search folder for any filename that contains the index
    # as a standalone token (prevents matching digits inside larger numbers).
    try:
        token_re = re.compile(rf"(?<!\\d){re.escape(str(idx))}(?!\\d)")
        for fname in os.listdir(folder):
            _, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            name = os.path.splitext(fname)[0]
            if token_re.search(name):
                return os.path.join(folder, fname)
    except Exception:
        pass

    return None


def parse_json_loose(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _extract_message_content(resp_json):
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        pass
    err = resp_json.get("error") if isinstance(resp_json, dict) else None
    if isinstance(err, dict):
        msg = err.get("message")
        meta = err.get("metadata")
        if isinstance(meta, dict) and meta.get("raw"):
            return f"API_ERROR: {msg} | raw={meta.get('raw')}"
        return f"API_ERROR: {msg}"
    return "API_ERROR: malformed response without choices"


def _post_chat_and_extract_content(payload, timeout=120):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    status = r.status_code
    try:
        resp_json = r.json()
    except Exception:
        text = (r.text or "").strip()
        snippet = text[:600] if text else "<empty body>"
        raise ValueError(f"HTTP_{status}: non-JSON response: {snippet}")

    content = _extract_message_content(resp_json)
    if content.startswith("API_ERROR:"):
        raise ValueError(f"HTTP_{status}: {content}")
    return content


def call_model_for_ars(b64_image, mime_type):
    payload = {
        "model": MODEL,
        "max_tokens": 3000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                    {"type": "text", "text": "Generate ARS with type labels as specified."}
                ],
            },
        ],
    }
    return _post_chat_and_extract_content(payload, timeout=120)


def validate_perception_only(b64_image, mime_type, ars_perception):
    prompt = (
        '''Given these perception-only sub-questions (each answer MUST be a single token: a number or a single-word label). DO NOT answer 'yes' or 'no'. If the relation would naturally be answered yes/no, instead supply a descriptive label (examples: parallel, perpendicular, collinear, intersecting, inside, outside, greater, equal).

'''
        + json.dumps(ars_perception, indent=2)
        + '''

Tasks:
1) Answer each sub-question using the image only (or text if depends_on_text == Yes). Each answer must be a single token: either a numeric literal (e.g., 45) or a lowercase alphabetic label with no spaces or punctuation (e.g., parallel).
2) Return strict JSON ONLY with these keys: { "valid": true/false, "sub_answers": {...}, "checks": { "all_answerable": true/false, "no_yes_no": true/false } }.

Rules for answers:
- NEVER use the tokens 'yes' or 'no'. If you would otherwise answer yes/no, choose a single-word label from the examples or a similarly descriptive single word.
- Labels must contain only letters (a-z) and no punctuation or whitespace.

Set valid=true only if all questions are answerable from provided inputs and no answers are 'yes'/'no'. Set "checks.no_yes_no" accordingly.
Respond with JSON only.
'''
    )
    payload = {
        "model": MODEL,
        "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }
    try:
        content = _post_chat_and_extract_content(payload, timeout=120)
        parsed = parse_json_loose(content)
        return parsed if isinstance(parsed, dict) else {"valid": False}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _clean_degrees_in_answers(sub_answers):
    if not isinstance(sub_answers, dict):
        return sub_answers
    cleaned = {}
    for k, v in sub_answers.items():
        if isinstance(v, str):
            v2 = v.replace("\\u00b0", "").replace("°", "").strip()
            v2 = re.sub(r"\s*degrees?\b", "", v2, flags=re.I).strip()
            cleaned[k] = v2
        else:
            cleaned[k] = v
    return cleaned


def filter_perception_only(ars_raw):
    if not isinstance(ars_raw, dict):
        return {}
    perception_ids = {qid for qid, obj in ars_raw.items()
                      if isinstance(obj, dict) and str(obj.get("type", "")).strip().lower() == "perception"}
    filtered = {}
    for qid in sorted(ars_raw.keys(), key=lambda x: x):
        if qid in perception_ids:
            obj = dict(ars_raw[qid])
            deps = obj.get("depends_on_sub_question", [])
            if isinstance(deps, list):
                obj["depends_on_sub_question"] = [d for d in deps if d in perception_ids]
            filtered[qid] = obj
    return filtered


def process_index(idx):
    image_path = find_image_by_index(IMAGE_FOLDER, idx)
    if not image_path:
        return {"status": "IMAGE_NOT_FOUND", "ars": {}}
    b64 = encode_image(image_path)
    mime_type = detect_image_mime(image_path)
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_model_for_ars(b64, mime_type)
            ars_raw = parse_json_loose(raw)
            if not ars_raw:
                raise ValueError("Empty or unparsable ARS")
            ars_perception = filter_perception_only(ars_raw)
            if not ars_perception:
                raise ValueError("No perception-type questions produced")
            validation = validate_perception_only(b64, mime_type, ars_perception)
            if validation.get("valid"):
                if isinstance(validation.get("sub_answers"), dict):
                    validation["sub_answers"] = _clean_degrees_in_answers(validation["sub_answers"])
                return {"status": "OK", "ars": ars_perception, "validation": validation}
            raise ValueError("Validation failed")
        except Exception as e:
            print(f"[Attempt {attempt}] Index {idx} -> {e}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= RETRY_BACKOFF
    return {"status": "FAILED"}


def main():
    if not API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY")
    # Load existing results to avoid re-generating ARS for already-processed questions
    results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    def _already_generated(entry):
        if not entry:
            return False
        if isinstance(entry, dict):
            status = str(entry.get("status", "")).upper()
            ars = entry.get("ars")
            if status == "OK":
                return True
            if isinstance(ars, dict) and len(ars) > 0:
                return True
        return False

    for idx in TARGET_PROBLEMS:
        print(f"Processing index: {idx}")
        # handle JSON keys that might be strings
        existing = None
        if str(idx) in results:
            existing = results.get(str(idx))
        elif idx in results:
            existing = results.get(idx)

        if existing and _already_generated(existing):
            print(f"Skipping index {idx}: ARS already generated.")
            # normalize stored result under the integer key
            results[idx] = existing
            continue

        results[idx] = process_index(idx)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Generate a simple report: total questions and subquestion counts per question
    report = {}
    sub_counts = {}
    total_with_ars = 0
    for k, v in results.items():
        try:
            qid = int(k)
        except Exception:
            qid = k
        ars = None
        if isinstance(v, dict):
            ars = v.get("ars")
        count = len(ars) if isinstance(ars, dict) else 0
        sub_counts[str(qid)] = count
        if count > 0:
            total_with_ars += 1

    report["total_questions_in_results"] = len(results)
    report["total_questions_with_ars"] = total_with_ars
    report["subquestion_counts"] = sub_counts

    try:
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report written to {REPORT_JSON}: {total_with_ars} questions with ARS")
    except Exception as e:
        print(f"Failed to write report: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
