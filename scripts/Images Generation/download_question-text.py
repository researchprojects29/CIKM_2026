import csv
from datasets import load_dataset

# TARGET_INDICES = [
#     1,2,3,4,7,8,9,10,11,16,17,20,22,25,26,29,30,112,117,119,133,149,198,207,210,211,239,243,348,432,
#     34, 36, 37, 38, 40, 43, 45, 46, 48, 49, 50, 53, 55, 56, 57, 58, 59,
#     60, 72, 73, 78, 79, 81, 84, 86, 88, 109, 123, 150, 165 
# ]
TARGET_INDICES = [420]
OUTPUT_TXT = "questions-anshuetvihaan.txt"
OUTPUT_CSV = "questions-anshuetvihaan.csv"


def load_questions(target_indices: list) -> dict:
    """
    Streams testmini_text_only from HuggingFace.
    Returns { problem_index (int): question (str) }
    """
    print("Loading testmini_text_only from HuggingFace (streaming)...")
    dataset = load_dataset(
        "AI4Math/MathVerse",
        "testmini_text_only",
        split="testmini_text_only",
        streaming=True
    )

    target_set = set(str(i) for i in target_indices)
    collected  = {}

    for sample in dataset:
        pid = sample["problem_index"]
        if pid in target_set and int(pid) not in collected:
            collected[int(pid)] = sample["question"]
            if len(collected) == len(target_indices):
                break

    missing = set(target_indices) - set(collected.keys())
    if missing:
        print(f"[Warning] Could not find questions for indices: {sorted(missing)}")

    print(f"Loaded {len(collected)} questions.\n")
    return collected


def save_txt(questions: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for idx in sorted(questions.keys()):
            f.write(f"[Problem Index: {idx}]\n")
            f.write(f"{questions[idx]}\n")
            f.write("-" * 60 + "\n\n")
    print(f"Saved: {filepath}")


def save_csv(questions: dict, filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_index", "question"])
        writer.writeheader()
        for idx in sorted(questions.keys()):
            writer.writerow({"problem_index": idx, "question": questions[idx]})
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    questions = load_questions(TARGET_INDICES)
    save_txt(questions, OUTPUT_TXT)
    save_csv(questions, OUTPUT_CSV)