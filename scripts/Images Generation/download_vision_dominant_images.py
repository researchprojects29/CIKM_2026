# import os
# from io import BytesIO

# import requests
# from datasets import load_dataset
# from PIL import Image


# TARGET_INDICES = [
#     14, 52, 64, 84, 95, 99, 107, 108, 115, 120,
#     132, 136, 151, 152, 153, 157, 223, 257, 264,
#     281, 298, 364, 365, 438, 441, 447, 449, 457, 472, 779
# ]

# OUTPUT_DIR = "vision_dominant_images"
# DATASET_NAME = "AI4Math/MathVerse"

# # Common config candidates used by MathVerse variants.
# CONFIG_CANDIDATES = [
#     # "testmini_vision_dominant",
#     "vision_dominant",
#     # "testmini",
# ]


# def get_image_obj(sample):
#     """Returns a PIL image from a sample image field."""
#     image_field = sample.get("image")

#     if image_field is None:
#         raise ValueError("Sample has no 'image' field.")

#     if isinstance(image_field, Image.Image):
#         return image_field.convert("RGB")

#     if isinstance(image_field, dict):
#         # If datasets decoded image exists.
#         if image_field.get("bytes"):
#             return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
#         # If only path/url is provided.
#         if image_field.get("path"):
#             path_or_url = image_field["path"]
#             if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
#                 resp = requests.get(path_or_url, timeout=30)
#                 resp.raise_for_status()
#                 return Image.open(BytesIO(resp.content)).convert("RGB")
#             return Image.open(path_or_url).convert("RGB")

#     if isinstance(image_field, str):
#         if image_field.startswith("http://") or image_field.startswith("https://"):
#             resp = requests.get(image_field, timeout=30)
#             resp.raise_for_status()
#             return Image.open(BytesIO(resp.content)).convert("RGB")
#         return Image.open(image_field).convert("RGB")

#     raise TypeError(f"Unsupported image type: {type(image_field)}")


# def try_load_split():
#     """Tries likely vision-dominant configs and returns a streaming dataset."""
#     for config in CONFIG_CANDIDATES:
#         try:
#             print(f"Trying config: {config}")
#             ds = load_dataset(
#                 DATASET_NAME,
#                 config,
#                 split=config,
#                 streaming=True,
#             )
#             return ds, config
#         except Exception:
#             continue
#     raise RuntimeError(
#         "Could not load a vision-dominant split. "
#         "Check available configs for AI4Math/MathVerse."
#     )


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     target_set = {str(i) for i in TARGET_INDICES}
#     saved = set()

#     dataset, used_config = try_load_split()
#     print(f"Using split/config: {used_config}")

#     for sample in dataset:
#         pid = str(sample.get("problem_index", ""))
#         if pid not in target_set or int(pid) in saved:
#             continue

#         try:
#             img = get_image_obj(sample)
#             out_path = os.path.join(OUTPUT_DIR, f"{pid}.jpg")
#             img.save(out_path, format="JPEG", quality=95)
#             saved.add(int(pid))
#             print(f"Saved {out_path}")
#         except Exception as err:
#             print(f"[Error] problem_index={pid}: {err}")

#         if len(saved) == len(TARGET_INDICES):
#             break

#     missing = sorted(set(TARGET_INDICES) - saved)
#     print(f"\nDone. Saved {len(saved)} images to '{OUTPUT_DIR}'.")
#     if missing:
#         print(f"Missing indices: {missing}")


# if __name__ == "__main__":
#     main()

import os
from datasets import load_dataset

# TARGET_INDICES = [
#     1,2,3,4,7,8,9,10,11,16,17,20,22,25,26,29,30,112,117,119,133,149,198,207,210,211,239,243,348,432,
#     34, 36, 37, 38, 40, 43, 45, 46, 48, 49, 50, 53, 55, 56, 57, 58, 59,
#     60, 72, 73, 78, 79, 81, 84, 86, 88, 109, 123, 150, 165
# ]
TARGET_INDICES = [420]
OUTPUT_DIR = "./vision_dominant_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MathVerse testmini from HuggingFace...")
raw = load_dataset("AI4Math/MathVerse", "testmini")
dataset = raw["testmini"]

print(f"Dataset loaded. Total samples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

target_set = set(str(i) for i in TARGET_INDICES)
saved = {}

for sample in dataset:
    pid     = sample["problem_index"]
    version = sample["problem_version"]

    if pid in target_set and version == "Vision Dominant" and int(pid) not in saved:
        pil_image = sample["image"]
        filepath  = os.path.join(OUTPUT_DIR, f"{pid}.jpg")
        pil_image.convert("RGB").save(filepath, format="JPEG")
        saved[int(pid)] = filepath
        print(f"  Saved [{len(saved)}/{len(TARGET_INDICES)}]: {filepath}")

        if len(saved) == len(TARGET_INDICES):
            break

missing = set(TARGET_INDICES) - set(saved.keys())
if missing:
    print(f"\n[Warning] No 'Vision Dominant' image found for indices: {sorted(missing)}")

print(f"\nDone. {len(saved)} images saved to: {OUTPUT_DIR}/")