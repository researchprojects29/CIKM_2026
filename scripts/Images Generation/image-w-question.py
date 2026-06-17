import os
import csv
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL   = "google/gemini-3-pro-image-preview"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGES_DIR  = "./vision_dominant_images"   # folder with {index}.jpg files
CSV_PATH    = "./questions-anshuetvihaan.csv"            # columns: problem_index, question
OUTPUT_DIR  = "./annotated_images"

RERUN_INDICES = [420]

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_base64_image(b64_data: str, filepath: str):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))

def load_questions(csv_path: str) -> dict:
    """Loads { problem_index (int): question (str) } from the CSV."""
    questions = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions[int(row["problem_index"])] = row["question"].strip()
    return questions


# ── Core generation call ──────────────────────────────────────────────────────

def generate_annotated_image(b64_image: str, question: str, output_path: str) -> bool:
    """
    Sends the original diagram + question text to Gemini.
    Asks it to produce a self-sufficient annotated image:
      - Original diagram preserved exactly
      - A visual annotation (e.g. "?") placed on the diagram marking what to find
      - The full question text added as a caption below the diagram
    Saves the result to output_path.
    Returns True on success, False on failure.
    """
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    prompt = (
        "You are given a geometry diagram and a math question. "
        "Your task is to produce a single, self-sufficient image that combines both, "
        "exactly like a textbook problem image. Follow these rules strictly:\n\n"

        "1. PRESERVE the original diagram exactly — do not alter any lines, labels, "
        "numbers, angles, orientation of the diagram, or geometric structures and make sure you do not miss/crop any part of the original diagram.\n"

        "2. ADD a visual annotation directly on the diagram to mark what quantity "
        "is being asked for. For example:\n"
        "   - If the question asks for an angle, mark the angle and annotate it with '?'.\n"
        "   - If it asks for a side length, write length of $side_name = ?\n"
        "   - If it asks for an area, write 'Area of = ?' inside or beside the shape.\n"
        "   - Keep the annotation minimal and consistent with the diagram's style.\n\n"

        "3. ADD the full question text as a caption below the diagram, "
        "in clean readable font, exactly as provided — do not paraphrase or shorten it. \n"
        "Do not overlap the question text with the diagram.\n\n"

        "4. The final image should look like a clean, professional geometry problem "
        "as it would appear in a math textbook or exam paper.\n\n"

        f"Question text to use as caption:\n{question}"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mathverse-noise-injector.com",
        "X-Title": "MathVerse Image Annotator"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }
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
                print(f"      [Error] No image returned. Response: {result}")
        return False

    except Exception as e:
        print(f"      [API Error] {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load questions from CSV
    questions = load_questions(CSV_PATH)
    print(f"Loaded {len(questions)} questions from {CSV_PATH}")

    RERUN_INDICES = [420]

    filtered = {k: v for k, v in questions.items() if not RERUN_INDICES or k in RERUN_INDICES}
    total   = len(filtered)
    success = 0
    failed  = []

    for i, (idx, question) in enumerate(sorted(filtered.items()), start=1):

        image_path  = os.path.join(IMAGES_DIR, f"{idx}.jpg")
        output_path = os.path.join(OUTPUT_DIR, f"{idx}.jpg")

        print(f"\n[{i}/{total}] Problem index {idx}")

        # Check image exists
        if not os.path.exists(image_path):
            print(f"  [Skip] Image not found: {image_path}")
            failed.append(idx)
            continue

        # Encode image
        b64_image = encode_image(image_path)

        # Generate annotated image
        print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"  Calling Gemini...")
        ok = generate_annotated_image(b64_image, question, output_path)

        if ok:
            print(f"  -> Saved: {output_path}")
            success += 1
        else:
            print(f"  -> FAILED")
            failed.append(idx)

    # Summary
    print(f"\n{'='*50}")
    print(f"Done. {success}/{total} images generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}/")
    if failed:
        print(f"Failed indices: {failed}")
    print(f"{'='*50}")