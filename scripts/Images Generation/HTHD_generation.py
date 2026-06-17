import os
import csv
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY     = os.getenv("OPENROUTER_API_KEY")
API_URL     = "https://openrouter.ai/api/v1/chat/completions"
SPLIT_MODEL = "anthropic/claude-sonnet-4-5"
EDIT_MODEL  = "google/gemini-3-pro-image-preview"

IMAGES_DIR   = "./vision_dominant_images"   # original {index}.jpg files
CSV_PATH     = "./questions.csv"            # columns: problem_index, question
OUTPUT_DIR   = "./cross_reference_images"   # final enumerated images
INTERIM_DIR  = "./cross_reference_interim"  # edited diagrams before enumeration
SPLIT_LOG    = "./split_log.csv"

TARGET_INDICES = [108]
# [11,16,17,20,22,25,26,29,30,112,117,119,133,149,198,207,210,211,239,243,348,432,
#     34, 36, 37, 38, 40, 43, 45, 46, 48, 49, 50, 53, 55, 56, 57, 58, 59,
#     60, 72, 73, 78, 79, 81, 84, 86, 88, 109, 123, 150, 165 ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def b64_from_url(image_url):
    """Strips data URI prefix and returns raw base64 string."""
    if "," in image_url:
        return image_url.split(",", 1)[1]
    return image_url

def save_base64_image(b64_data, filepath):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))

def load_questions(csv_path):
    questions = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["problem_index"])
            if idx in TARGET_INDICES:
                questions[idx] = row["question"].strip()
    return questions

def init_split_log(filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem_index", "original_question", "modified_question",
            "removed_from_text", "removed_from_diagram"
        ])
        writer.writeheader()

def append_split_log(filepath, row):
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem_index", "original_question", "modified_question",
            "removed_from_text", "removed_from_diagram"
        ])
        writer.writerow(row)


# ── Step 1: Claude decides the split ─────────────────────────────────────────

def decide_split(question, b64_image):
    """
    Uses Claude to decide a bidirectional split:
    - Some info removed from text (still visible in diagram)
    - Some info removed from diagram (still present in text)
    Both sides MUST lose something essential.
    """

    user_prompt = (
        "You are an expert at analyzing geometry math problems.\n\n"

        "TASK: You will receive a geometry question text and its diagram. "
        "Your job is to create a BIDIRECTIONAL split of the mathematical "
        "information — meaning BOTH the text and the diagram must each lose "
        "at least one piece of essential numerical or geometric information, "
        "and that missing piece must be present only in the other modality. "
        "A solver MUST look at both the text and the diagram together to solve "
        "the problem. Neither alone should be sufficient.\n\n"

        "STRICT RULES:\n"
        "1. You MUST remove at least one specific numerical value or geometric "
        "condition from the question text. The removed value must be clearly "
        "visible as a label or annotation in the diagram.\n"
        "2. You MUST also identify at least one specific numerical value or "
        "label that is visible in the diagram and must be erased from the "
        "diagram. That value must still be present in the (modified) question text. Vice versa also stands true.\n"
        "For example, if you remove the measure of angle A = 30° from the text, you must still be able to see it in the diagram and vice versa. Either one of the modalitites should have the information to solve the problem, it should not be removed from both the modalities.\n"
        "3. You can remove the final question being asked (e.g. 'find angle X', "
        "'what is the length of BC') either from the text or the diagram but make sure it is still present in one of the other modalities. \n"
        "For example, if you remove the question from the text, you must still be able to see it in the diagram and vice versa.\n"
        "4. Do NOT add any new information to either the text or the diagram.\n"
        "5. The removed pieces must be genuinely necessary for solving — not "
        "trivial labels like point names (A, B, C) or the shape type.\n"
        "6. If the diagram has no numerical labels at all, then for rule 2, "
        "identify a geometric relationship visible in the diagram (e.g. a right "
        "angle mark, parallel line arrows, tick marks for equal sides) and "
        "describe erasing that marking.\n\n"

        "IMPORTANT — diagram_edit_instruction must be extremely specific:\n"
        "- State exactly which label/value/marking to erase\n"
        "- State exactly where it is located in the diagram (e.g. 'the number "
        "35 near vertex A at the bottom-left of the circle')\n"
        "- Say to replace it with clean background — do not leave any residue\n\n"

        f"Question text:\n{question}\n\n"

        "Respond with ONLY a raw JSON object using exactly these four keys:\n"
        "modified_question, removed_from_text, removed_from_diagram, "
        "diagram_edit_instruction"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mathverse-noise-injector.com",
        "X-Title": "MathVerse Cross-Reference Generator"
    }

    payload = {
        "model": SPLIT_MODEL,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "{"    # prefill forces raw JSON output
            }
        ]
    }

    raw_text = ""
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result   = response.json()
        raw_text = result["choices"][0]["message"]["content"].strip()

        if not raw_text.startswith("{"):
            raw_text = "{" + raw_text

        last = raw_text.rfind("}")
        if last != -1:
            raw_text = raw_text[:last + 1]

        parsed = json.loads(raw_text)

        missing_text    = parsed.get("removed_from_text", "").strip()
        missing_diagram = parsed.get("removed_from_diagram", "").strip()

        if not missing_text or missing_text.lower() in ("none", "nothing", "n/a", "-"):
            print(f"      [Warning] Nothing removed from text — split may be one-sided.")
        if not missing_diagram or missing_diagram.lower() in ("none", "nothing", "n/a", "-"):
            print(f"      [Warning] Nothing removed from diagram — split may be one-sided.")

        return parsed

    except json.JSONDecodeError as e:
        print(f"      [Split Error] Could not parse JSON: {e}")
        print(f"      Raw response: {raw_text[:300]}")
        return None
    except Exception as e:
        print(f"      [Split API Error] {e}")
        return None


# ── Step 2: Gemini erases labels from diagram ─────────────────────────────────

def edit_diagram(b64_image, edit_instruction, interim_path):
    """
    Erases specific labels from the diagram.
    Saves the edited diagram to interim_path.
    Returns the base64 of the edited image (for Step 3), or None on failure.
    """

    prompt = (
        "You are given a geometry diagram. Your task is to ERASE specific "
        "labels or markings from this diagram as described below. "
        "Everything else in the diagram must remain completely unchanged.\n\n"
        "WHAT TO ERASE:\n"
        f"{edit_instruction}\n\n"
        "STRICT RULES:\n"
        "1. Do NOT alter any lines, shapes, angles, or geometric structures.\n"
        "2. Do NOT change, move, or remove any labels other than those "
        "explicitly listed above.\n"
        "3. Do NOT add any new labels, annotations, or information.\n"
        "4. Where you erase a label, fill the area with clean background "
        "(white or matching the surrounding color) — no smudges or residue.\n"
        "5. The diagram must look identical to the original in every way "
        "except for the erased items."
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mathverse-noise-injector.com",
        "X-Title": "MathVerse Cross-Reference Generator"
    }

    payload = {
        "model": EDIT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            }
        ],
        "modalities": ["image", "text"]
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if "choices" in result and result["choices"]:
            message = result["choices"][0]["message"]
            if "images" in message and message["images"]:
                image_url  = message["images"][0]["image_url"]["url"]
                edited_b64 = b64_from_url(image_url)
                save_base64_image(edited_b64, interim_path)
                return edited_b64
            else:
                print(f"      [Edit Error] No image returned.")
        return None

    except Exception as e:
        print(f"      [Edit API Error] {e}")
        return None


# ── Step 3: Gemini combines edited diagram + modified question text ────────────

def enumerate_image(edited_b64, modified_question, output_path):
    """
    Takes the edited diagram (labels erased) and the modified question text,
    and produces a single self-sufficient image:
      - Edited diagram on top, preserved exactly
      - Modified question text as a clean caption below
      - A '?' annotation placed on the diagram at the quantity being asked for
    Saves the final image to output_path.
    Returns True on success, False on failure.
    """

    prompt = (
        "You are given a geometry diagram and a question text. "
        "Your task is to produce a single, self-sufficient image that combines "
        "both, exactly like a textbook problem image.\n\n"

        "RULES:\n"
        "1. PRESERVE the diagram exactly as given — do not alter any lines, "
        "labels, numbers, angles, or structures. The diagram has already been "
        "edited — do not undo or add back anything.\n"
        "2. ADD an annotation like '?' or 'length of BC = ?' directly on or near the diagram to mark what quantity "
        "is being asked for. For example:\n"
        "   - If the question asks for an angle, place '?' at that angle.\n"
        "   - If it asks for a side length, place '?' beside that side.\n"
        "   - If it asks for an area, write 'Area = ?' inside or beside the shape.\n"
        "   Keep the annotation minimal and consistent with the diagram style.\n"
        "3. ADD the question text as a caption below the diagram in clean, "
        "readable font — exactly as provided, word for word. Do not paraphrase "
        "or shorten it.\n"
        "Do NOT overlap the question text with the diagram.\n"
        "4. The final image should look like a clean, professional geometry "
        "problem as it would appear in a math textbook or exam paper.\n"
        "5. Do NOT add any other text or information beyond the caption and "
        "the '?' marker.\n\n"
        f"Question text to use as caption:\n{modified_question}"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mathverse-noise-injector.com",
        "X-Title": "MathVerse Cross-Reference Generator"
    }

    payload = {
        "model": EDIT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{edited_b64}"}
                    }
                ]
            }
        ],
        "modalities": ["image", "text"]
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if "choices" in result and result["choices"]:
            message = result["choices"][0]["message"]
            if "images" in message and message["images"]:
                image_url = message["images"][0]["image_url"]["url"]
                save_base64_image(image_url, output_path)
                return True
            else:
                print(f"      [Enumerate Error] No image returned.")
        return False

    except Exception as e:
        print(f"      [Enumerate API Error] {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    init_split_log(SPLIT_LOG)

    questions = load_questions(CSV_PATH)
    print(f"Loaded {len(questions)} questions.\n")

    total   = len(questions)
    success = 0
    failed  = []

    for i, idx in enumerate(sorted(questions.keys()), start=1):

        question     = questions[idx]
        image_path   = os.path.join(IMAGES_DIR,  f"{idx}.jpg")
        interim_path = os.path.join(INTERIM_DIR, f"{idx}.jpg")
        output_path  = os.path.join(OUTPUT_DIR,  f"{idx}.jpg")

        print(f"\n[{i}/{total}] Problem index {idx}")

        if not os.path.exists(image_path):
            print(f"  [Skip] Image not found: {image_path}")
            failed.append(idx)
            continue

        b64_image = encode_image(image_path)

        # ── Step 1: Decide the split ───────────────────────────────────────
        print(f"  Step 1: Deciding information split (Claude)...")
        split = decide_split(question, b64_image)

        if split is None:
            print(f"  [Failed] Could not determine split.")
            failed.append(idx)
            continue

        modified_question = split.get("modified_question", question)
        print(f"  Removed from text    : {split.get('removed_from_text', '')}")
        print(f"  Removed from diagram : {split.get('removed_from_diagram', '')}")

        # ── Step 2: Erase labels from diagram ─────────────────────────────
        print(f"  Step 2: Erasing labels from diagram (Gemini)...")
        edited_b64 = edit_diagram(
            b64_image,
            split.get("diagram_edit_instruction", ""),
            interim_path
        )

        if edited_b64 is None:
            print(f"  [Failed] Diagram edit failed.")
            failed.append(idx)
            continue

        print(f"  -> Interim saved: {interim_path}")

        # ── Step 3: Combine edited diagram + modified question text ────────
        print(f"  Step 3: Enumerating image with modified question (Gemini)...")
        ok = enumerate_image(edited_b64, modified_question, output_path)

        if ok:
            print(f"  -> Final image saved: {output_path}")
            success += 1
        else:
            print(f"  [Failed] Enumeration failed.")
            failed.append(idx)

        # ── Log the split ──────────────────────────────────────────────────
        append_split_log(SPLIT_LOG, {
            "problem_index":        idx,
            "original_question":    question,
            "modified_question":    modified_question,
            "removed_from_text":    split.get("removed_from_text", ""),
            "removed_from_diagram": split.get("removed_from_diagram", "")
        })

    print(f"\n{'='*55}")
    print(f"Done. {success}/{total} images generated successfully.")
    print(f"Final images    : {OUTPUT_DIR}/")
    print(f"Interim diagrams: {INTERIM_DIR}/")
    print(f"Split log       : {SPLIT_LOG}")
    if failed:
        print(f"Failed indices  : {failed}")
    print(f"{'='*55}")