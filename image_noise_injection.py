import json
import base64
import os
import random
import re
import requests
 
API_KEY = "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"
MODEL_ID = "google/gemini-3-pro-image-preview"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
 
INPUT_DIR = r"ftfd"
OUTPUT_ROOT = "noisy_images"

# ---------------------------------------------------
# OPTIONAL SELECTION FILTERS
# Use only one, or both as None to process all images
# ---------------------------------------------------

# Example: {14, 32, 44}
SELECTED_INDICES = {117}
# Example: {"14.png", "32.jpg", "44.jpg"}
SELECTED_FILENAMES = None
 
 
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
 
 
def save_base64_image(b64_data, filepath):
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
 
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(b64_data))
 
 
def call_gemini_vision(image_input, prompt, filename):
    if not API_KEY or API_KEY == "YOUR_OPENROUTER_API_KEY":
        raise ValueError("Please set your real OpenRouter API key in API_KEY.")
 
    if os.path.isfile(image_input):
        b64_image = encode_image(image_input)
    else:
        b64_image = image_input
 
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mathverse-noise-injector.com",
        "X-Title": "MathVerse Noise Injector"
    }
 
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a specialized image augmentation assistant for a math dataset. "
                            "Your task is to take the provided geometry diagram and RECREATE it "
                            "while applying a specific visual modification. "
                            "CRITICAL: Unless explicitly told to change values, you MUST preserve "
                            "all original numbers, variable labels, and geometric structures exactly. "
                            f"Task: {prompt}"
                        )
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
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        result = response.json()
 
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
 
            if "images" in message and len(message["images"]) > 0:
                image_url = message["images"][0]["image_url"]["url"]
                save_base64_image(image_url, filename)
                print(f"[OK] Saved to {filename}")
                return True
 
        print(f"[WARN] No image found in response for {filename}")
        return False
 
    except Exception as e:
        print(f"[FAIL] API request failed for {filename}: {e}")
        return False
 
 
# ---------------------------------------------------
# Noise techniques
# ---------------------------------------------------
 
def change_background(image_path, filename):
    strategies = [
        "Draw this diagram on a sheet of graph paper (grid background). Keep lines dark and distinct.",
        "Draw this diagram on a piece of crumpled, wrinkled white paper.",
        "Draw this diagram as if it were drawn with white chalk on a dark green blackboard."
    ]
    selected_prompt = random.choice(strategies)
    return call_gemini_vision(image_path, selected_prompt, filename)
 
 
def simulate_uneven_illumination(image_path, filename):
    prompt = (
        "Recreate this diagram exactly, but simulate uneven lighting. "
        "Cast a realistic, soft shadow across the top-right corner of the image, "
        "as if a hand or object is blocking the light. Ensure the diagram remains legible underneath."
    )
    return call_gemini_vision(image_path, prompt, filename)
 
 
def insert_irrelevant_objects(image_path, filename):
    objects = [
        "Place a realistic circular coffee mug stain on the paper, slightly overlapping a non-critical part of the diagram.",
        "Place a yellow wooden pencil lying diagonally across the empty space of the paper.",
        "Show a paperclip resting on the side of the diagram."
    ]
    selected_prompt = random.choice(objects)
    return call_gemini_vision(image_path, selected_prompt, filename)
 
def pixel_level_noise(image_path, filename):
    """
    Applies pixel-level noise using API.
    Randomly chooses between Gaussian noise and Salt & Pepper noise.
    """
 
    noise_type = random.choice(["gaussian", "salt_pepper"])
 
    if noise_type == "gaussian":
        prompt = (
            "Recreate this geometry diagram exactly, but add Gaussian noise across the entire image. "
            "The noise should appear as fine grain (sensor noise) while keeping all text, numbers, "
            "and diagram elements readable. Do NOT change any labels, values, or structure."
        )
 
    else:  # salt & pepper
        prompt = (
            "Recreate this geometry diagram exactly, but add salt and pepper noise. "
            "Introduce small random black and white pixels across the image, simulating impulse noise. "
            "Ensure the diagram, labels, and numbers remain readable and unchanged."
        )
 
    return call_gemini_vision(image_path, prompt, filename)
 
 
def blur(image_path, filename):
    prompt = (
        "Recreate this geometry diagram exactly, but blur the image. "
        "Ensure ALL text, numbers, and lines must remain readable. "
        "Do NOT over-blur or distort the diagram structure."
    )
    return call_gemini_vision(image_path, prompt, filename)
 
def make_disproportionate(image_path, filename):
    prompt = (
        "Redraw this geometry diagram but distort the visual proportions. "
        "Stretch the shapes horizontally or vertically so that they look visually "
        "disproportionate to their labels (e.g., a square looking like a rectangle). "
        "CRITICAL: Keep the text labels and numbers EXACTLY as they are in the original."
    )
    return call_gemini_vision(image_path, prompt, filename)
 
 
# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

_QNUM_RE = re.compile(r"\bQ\s*0*(\d+)\b", re.IGNORECASE)
_PLAIN_NUM_RE = re.compile(r"^\s*0*(\d+)\s*$")
_ANY_NUM_RE = re.compile(r"(\d+)")

def extract_index_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    normalized = stem.replace("_", " ").replace("-", " ")

    m = _QNUM_RE.search(normalized)
    if m:
        return int(m.group(1))

    m = _PLAIN_NUM_RE.match(stem)
    if m:
        return int(m.group(1))

    nums = _ANY_NUM_RE.findall(stem)
    if len(nums) == 1:
        return int(nums[0])

    return None

def is_selected_image(filename):
    base_name = os.path.basename(filename)

    if SELECTED_FILENAMES is not None:
        return base_name in SELECTED_FILENAMES

    if SELECTED_INDICES is not None:
        idx = extract_index_from_filename(base_name)
        return idx is not None and idx in SELECTED_INDICES

    return True
 
def get_image_files(folder):
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(valid_exts) and is_selected_image(f)
    ]
 
 
def process_single_noise(noise_name, noise_function, input_dir=INPUT_DIR, output_root=OUTPUT_ROOT, seed=42):
    random.seed(seed)
 
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
 
    output_dir = os.path.join(output_root, noise_name)
    os.makedirs(output_dir, exist_ok=True)
 
    image_files = get_image_files(input_dir)
    print(f"\nProcessing {len(image_files)} images with noise: {noise_name}")
 
    success = 0
    failed = 0
 
    for image_path in image_files:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, base_name)
 
        if os.path.exists(output_path):
            print(f"[SKIP] {base_name}")
            success += 1
            continue
 
        result = noise_function(image_path, output_path)
        if result:
            success += 1
        else:
            failed += 1
 
    print(f"\nDone: {noise_name}")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Saved in: {output_dir}")
 
 
# ---------------------------------------------------
# Main: generate ALL noises
# ---------------------------------------------------
 
if __name__ == "__main__":
    noise_map = {
        # "background": change_background,
        # "illumination": simulate_uneven_illumination,
        # "irrelevant_objects": insert_irrelevant_objects,
        # "disproportionate": make_disproportionate,
        # "pixel_noise": pixel_level_noise,
        "blur": blur
    }

    if SELECTED_FILENAMES is not None:
        print(f"Filtering by filenames: {sorted(SELECTED_FILENAMES)}")
    elif SELECTED_INDICES is not None:
        print(f"Filtering by indices: {sorted(SELECTED_INDICES)}")
    else:
        print("Processing all images")
 
    for noise_name, noise_function in noise_map.items():
        process_single_noise(
            noise_name=noise_name,
            noise_function=noise_function,
            input_dir=INPUT_DIR,
            output_root=OUTPUT_ROOT,
            seed=42
        )
 
    print("\nAll noise folders generated successfully!")
 
 
 
 
 
 
 
 
### CATEGORY 2: Loss of relevant information (VLM is forced to look at other modality)
 
"""
def modify_image_values(image_path, filename):
   
    Changes numerical values (labels) in the image to violate constraints.
   
    prompt = (
        "Recreate this diagram, but I want you to CHANGE one of the numerical labels "
        "to a value that makes no geometric sense. "
        "For example, if there is a right angle, label it '100°', or if two sides look equal, "
        "label one '5' and the other '20'. Only change the text label; keep the drawing the same."
    )
    return call_gemini_vision(image_path, prompt, filename)
 
def occlude_critical_info(image_path, filename):
   
    Occludes critical parts of the diagram (Ink blot, Hand, Tear).
   
    occlusions = [
        "Overlay a large, messy black ink blot that completely covers one of the numbers or variables in the diagram.",
        "Show a human finger pointing at the diagram, but positioned such that it blocks the view of a critical angle or side length.",
        "Simulate a rip or tear in the paper that removes a section of the diagram containing a label."
    ]
    selected_prompt = random.choice(occlusions)
    return call_gemini_vision(image_path, selected_prompt, filename)
"""