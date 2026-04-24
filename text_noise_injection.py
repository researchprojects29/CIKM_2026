import random
import re
import nltk
import os

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download("wordnet", quiet=True)

# ---------------------------------------
# Constants
# ---------------------------------------

OP_TOKENS = {
    "=", "+", "-", "*", "/", "^", "%", "(", ")", "[", "]", "{", "}",
    "<", ">", "<=", ">=", "==", "!=", "~", "≈", "$", "\\", "_{", "}^"
}

# Updated for: [Problem Index: 14]
HEADER_RE = re.compile(r"^\s*\[Problem\s+Index:\s*(\d+)\]\s*$", re.IGNORECASE)

# ---------------------------------------
# OPTIONAL FILTERS
# Keep as None to process all questions
# ---------------------------------------

# Example: {14, 32, 44}
SELECTED_INDICES = {34, 36, 37, 38, 40, 43, 45, 46, 48, 49, 50, 53, 55, 56, 57, 58, 59, 60, 72, 73, 78, 79, 81, 86, 88, 109, 123, 150, 165, 420}

# ---------------------------------------
# Helper functions
# ---------------------------------------

def split_trailing_punct(tok: str):
    """Separates word from trailing punctuation while keeping decimals intact."""
    m = re.fullmatch(r"(.*?)([.,!?;:]*)", tok)
    return (m.group(1), m.group(2)) if m else (tok, "")

def is_operator(tok: str) -> bool:
    core, _ = split_trailing_punct(tok)
    return core in OP_TOKENS or any(op in tok for op in OP_TOKENS)

def is_number_token(tok: str) -> bool:
    """Checks for numbers, including decimals and comma-formatted numbers."""
    core, _ = split_trailing_punct(tok)
    return bool(re.fullmatch(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", core))

def contains_digit(tok: str) -> bool:
    return bool(re.search(r"\d", tok))

def protected_indices(tokens: list) -> set:
    """
    Protect tokens that are mathematical, numeric, or adjacent to them.
    """
    bad = set()
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if is_number_token(tok) or contains_digit(tok) or is_operator(tok):
            for j in (i - 1, i, i + 1):
                if 0 <= j < n:
                    bad.add(j)
    return bad

def extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+", text))

def sentence_boundaries(text: str) -> list:
    """Find safe sentence boundaries without breaking decimals."""
    boundaries = []
    for i, ch in enumerate(text):
        if ch not in ".!?":
            continue
        prev_ch = text[i - 1] if i > 0 else ""
        next_ch = text[i + 1] if i + 1 < len(text) else ""
        if prev_ch.isdigit() and next_ch.isdigit():
            continue
        boundaries.append(i + 1)

    return boundaries if boundaries else [len(text)]

def filter_question_pairs(question_pairs, selected_indices=None):
    """
    Filter loaded questions by problem index.
    If selected_indices is None, keep all.
    """
    if selected_indices is None:
        return question_pairs
    return [(qid, qtext) for qid, qtext in question_pairs if qid in selected_indices]

# ---------------------------------------
# Noise functions
# ---------------------------------------

def inject_typos(text, error_rate=0.4):
    """
    Randomly drops or swaps characters in non-math words.
    """
    tokens = text.split()
    bad = protected_indices(tokens)

    editable = []
    for i, t in enumerate(tokens):
        if i in bad:
            continue
        core, punct = split_trailing_punct(t)
        if len(core) >= 4 and core.isalpha():
            editable.append((i, core, punct))

    if not editable:
        return text

    num_errors = max(1, int(len(editable) * error_rate))

    for _ in range(num_errors):
        if not editable:
            break

        idx_in_editable = random.randrange(len(editable))
        i, core, punct = editable.pop(idx_in_editable)

        if len(core) < 2:
            continue

        if random.random() < 0.5:
            char_idx = random.randrange(len(core))
            new_core = core[:char_idx] + core[char_idx + 1:]
        else:
            if len(core) < 2:
                new_core = core
            else:
                char_idx = random.randrange(len(core) - 1)
                new_core = (
                    core[:char_idx]
                    + core[char_idx + 1]
                    + core[char_idx]
                    + core[char_idx + 2:]
                )

        tokens[i] = new_core + punct

    return " ".join(tokens)

def add_distractor_sentences(text):
    """
    Inserts at least two irrelevant sentences at valid sentence boundaries.
    Rules:
    - distractors are not inserted as the final sentence
    - at least 2 distractors are inserted when possible
    - no duplicate distractor sentence is used in the same question
    """
    distractors = [
        "The hallway felt unusually quiet.", 
        "A cup was resting near the window.",
        "He glanced at the clock and sighed.",
        "The curtain moved with the breeze."
        "She closed the notebook without a word.",
        "A bicycle was leaning against the wall.",
        "The phone vibrated once on the table.",
        "He pushed the chair back slowly.",
        "The room smelled faintly of rain.",
        "A paper slipped from the desk."
    ]

    existing_nums = extract_numbers(text)
    current_text = text

    valid_distractors = [
        d for d in distractors
        if not (extract_numbers(d) & existing_nums)
    ]

    if len(valid_distractors) >= 2:
        chosen_distractors = random.sample(valid_distractors, 2)
    elif len(valid_distractors) == 1:
        chosen_distractors = valid_distractors
    else:
        return text

    for sent in chosen_distractors:
        boundaries = sentence_boundaries(current_text)

        # distractor should never become the final sentence
        valid_boundaries = [b for b in boundaries if b < len(current_text)]

        if not valid_boundaries:
            break

        pos = random.choice(valid_boundaries)
        current_text = current_text[:pos] + " " + sent + " " + current_text[pos:]

    return current_text

def swap_punctuation(text):
    """
    Swaps '.' with ',' and vice versa, while protecting number-related tokens.
    """
    tokens = text.split()
    bad = protected_indices(tokens)

    new_tokens = []
    for i, tok in enumerate(tokens):
        if i in bad:
            new_tokens.append(tok)
        else:
            trans_table = str.maketrans({'.': ',', ',': '.'})
            new_tokens.append(tok.translate(trans_table))

    return " ".join(new_tokens)

def add_numeric_distractors(text):
    """
    Adds an irrelevant number phrase in a safe location.
    """
    tokens = text.split()
    existing_nums = extract_numbers(text)

    candidate = str(random.randint(10, 2025))
    while candidate in existing_nums:
        candidate = str(random.randint(10, 2025))

    distractor_phrases = [
        f"(Ref: {candidate})",
        f"[Page {candidate}]",
        f"ID:{candidate}"
    ]

    chosen = random.choice(distractor_phrases)
    bad = protected_indices(tokens)

    safe_slots = []
    for slot in range(len(tokens) + 1):
        left = slot - 1
        right = slot
        if (left >= 0 and left in bad) or (right < len(tokens) and right in bad):
            continue
        safe_slots.append(slot)

    if safe_slots:
        slot = random.choice(safe_slots)
        tokens.insert(slot, chosen)

    return " ".join(tokens)

# ---------------------------------------
# File loading / saving
# ---------------------------------------

def load_question_blocks_with_ids(txt_path):
    """
    Load questions from a file formatted like:
    [Problem Index: 14]
    text...

    Returns: [(14, "text..."), (32, "text..."), ...]
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    question_pairs = []
    cur_qid = None
    buf = []

    def flush():
        nonlocal cur_qid, buf
        if cur_qid is not None:
            text = "\n".join(buf).strip()
            if text:
                question_pairs.append((cur_qid, text))
        cur_qid = None
        buf = []

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            cur_qid = int(m.group(1))
        else:
            if cur_qid is not None:
                buf.append(line)

    flush()
    return question_pairs

def normalize_text(text):
    """
    Clean extra spaces but preserve line structure.
    """
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def save_questions(question_pairs, output_path):
    """
    Save questions preserving original IDs:
    [Problem Index: 14]
    question text
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for qid, qtext in question_pairs:
            qtext = normalize_text(qtext)
            f.write(f"[Problem Index: {qid}]\n")
            f.write(qtext + "\n\n")

# ---------------------------------------
# Noise application wrappers
# ---------------------------------------

def apply_typo_noise(question_pairs):
    return [(qid, inject_typos(q, error_rate=0.4)) for qid, q in question_pairs]

def apply_distractor_noise(question_pairs):
    return [(qid, add_distractor_sentences(q)) for qid, q in question_pairs]

def apply_punctuation_noise(question_pairs):
    return [(qid, swap_punctuation(q)) for qid, q in question_pairs]

def apply_numeric_noise(question_pairs):
    return [(qid, add_numeric_distractors(q)) for qid, q in question_pairs]

# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":
    input_file = r"Half_Text_Half_Diagram_Images\Half_text.txt"
    output_dir = r"noisy_questions_output"

    os.makedirs(output_dir, exist_ok=True)

    question_pairs = load_question_blocks_with_ids(input_file)
    print(f"Loaded {len(question_pairs)} questions")

    question_pairs = filter_question_pairs(question_pairs, SELECTED_INDICES)

    if SELECTED_INDICES is None:
        print("Processing all questions")
    else:
        print(f"Filtering by indices: {sorted(SELECTED_INDICES)}")

    print(f"Selected {len(question_pairs)} questions")

    random.seed(42)
    typo_questions = apply_typo_noise(question_pairs)
    save_questions(typo_questions, os.path.join(output_dir, "questions_typo.txt"))

    random.seed(42)
    distractor_questions = apply_distractor_noise(question_pairs)
    save_questions(distractor_questions, os.path.join(output_dir, "questions_distractor.txt"))

    random.seed(42)
    punctuation_questions = apply_punctuation_noise(question_pairs)
    save_questions(punctuation_questions, os.path.join(output_dir, "questions_punctuation.txt"))

    random.seed(42)
    numeric_questions = apply_numeric_noise(question_pairs)
    save_questions(numeric_questions, os.path.join(output_dir, "questions_numeric.txt"))

    print("Done. Saved all noisy files.")