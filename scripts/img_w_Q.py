from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

IMAGES_DIR = Path(r"noisy_images\blur")
QUESTIONS_FILE = Path(r"Half_Text_Half_Diagram_Images\Half_text.txt")
OUTPUT_DIR = Path(r"Half_text_Noisy_half_Diagram\blur_noise")

# Set to None to process all images
# Or give specific problem indices only
SELECTED_INDICES = {117}

BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"

PADDING_X_RATIO = 0.05
PADDING_Y_RATIO = 0.05
FOOTER_MIN_HEIGHT_RATIO = 0.18
FOOTER_MAX_HEIGHT_RATIO = 0.75
LINE_SPACING_MULT = 1.20
MIN_FONT_PX = 14

_HEADER_RE = re.compile(r"^\s*---\s*Question\s+(\d+)\s*---\s*$", re.IGNORECASE)
_PROBLEM_INDEX_RE = re.compile(r"^\s*\[Problem\s+Index:\s*(\d+)\]\s*$", re.IGNORECASE)
_QNUM_RE = re.compile(r"\bQ\s*0*(\d+)\b", re.IGNORECASE)
_PLAIN_NUM_RE = re.compile(r"^\s*0*(\d+)\s*$")


def extract_qnum_from_filename(name: str) -> Optional[int]:
    stem = Path(name).stem.strip()
    normalized = stem.replace("_", " ").replace("-", " ")

    m = _QNUM_RE.search(normalized)
    if m:
        return int(m.group(1))

    m = _PLAIN_NUM_RE.match(stem)
    if m:
        return int(m.group(1))

    return None


def load_questions_from_header_blocks(path: Path) -> Dict[int, str]:
    """
    Supports BOTH formats:

    Format 1:
        --- Question 34 ---
        question text...

    Format 2:
        [Problem Index: 34]
        question text...
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    out: Dict[int, str] = {}
    cur_qnum: Optional[int] = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal cur_qnum, buf
        if cur_qnum is None:
            return
        text = "\n".join(buf).strip()
        if text:
            out[cur_qnum] = text
        cur_qnum = None
        buf = []

    for ln in lines:
        m1 = _HEADER_RE.match(ln)
        m2 = _PROBLEM_INDEX_RE.match(ln)

        if m1:
            flush()
            cur_qnum = int(m1.group(1))
            continue

        if m2:
            flush()
            cur_qnum = int(m2.group(1))
            continue

        if cur_qnum is not None:
            buf.append(ln)

    flush()
    return out


def iter_image_files(images_dir: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def pick_font(font_px: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
        "/Library/Fonts/Arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_px)
        except Exception:
            continue
    return ImageFont.load_default()


def sanitize_question_text(text: str) -> str:
    return text.replace("∠", "")


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    cur = words[0]

    for w in words[1:]:
        trial = f"{cur} {w}"
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w

    lines.append(cur)
    return "\n".join(lines)


@dataclass(frozen=True)
class AnnotateResult:
    src: Path
    dst: Path
    qnum: int


def annotate_image_with_footer_text(src_path: Path, footer_text: str, dst_path: Path) -> None:
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    pad_x = max(12, int(w * PADDING_X_RATIO))
    pad_y = max(12, int(h * PADDING_Y_RATIO))
    footer_min_h = max(40, int(h * FOOTER_MIN_HEIGHT_RATIO))
    footer_max_h = int(h * FOOTER_MAX_HEIGHT_RATIO)

    scratch = Image.new("RGB", (w, h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(scratch)

    footer_text = sanitize_question_text(footer_text.strip())

    font_px = max(18, int(h * 0.050))
    max_text_width = max(1, w - 2 * pad_x)

    while True:
        font = pick_font(font_px)
        spacing = int(font_px * (LINE_SPACING_MULT - 1))
        wrapped = wrap_text(draw, footer_text, font, max_text_width)

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
        text_h = bbox[3] - bbox[1]
        needed_footer_h = max(footer_min_h, text_h + 2 * pad_y)

        if needed_footer_h <= footer_max_h or font_px <= MIN_FONT_PX:
            footer_h = min(needed_footer_h, footer_max_h)
            break

        font_px = int(font_px * 0.90)

    out = Image.new("RGB", (w, h + footer_h), BACKGROUND_COLOR)
    out.paste(img, (0, 0))

    out_draw = ImageDraw.Draw(out)
    spacing = int(font_px * (LINE_SPACING_MULT - 1))
    bbox2 = out_draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
    text_h2 = bbox2[3] - bbox2[1]

    footer_top = h
    text_y = footer_top + max(0, (footer_h - text_h2) // 2)

    out_draw.multiline_text(
        (pad_x, text_y),
        wrapped,
        font=font,
        fill=TEXT_COLOR,
        spacing=spacing,
        align="left",
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst_path)


def should_process(qnum: int) -> bool:
    return SELECTED_INDICES is None or qnum in SELECTED_INDICES


def main() -> None:
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Missing images dir: {IMAGES_DIR.resolve()}")
    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"Missing questions file: {QUESTIONS_FILE.resolve()}")

    questions = load_questions_from_header_blocks(QUESTIONS_FILE)
    if not questions:
        raise SystemExit(
            f"No questions parsed from: {QUESTIONS_FILE.resolve()}\n"
            "Expected headers like: --- Question 34 --- or [Problem Index: 34]"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    skipped = 0
    skipped_not_selected = 0

    found_selected_images = set()

    for img_path in iter_image_files(IMAGES_DIR):
        total += 1

        qnum = extract_qnum_from_filename(img_path.name)
        if qnum is None:
            skipped += 1
            print(f"[SKIP] {img_path.name} -> cannot extract Q-number from filename")
            continue

        if not should_process(qnum):
            skipped_not_selected += 1
            print(f"[SKIP] {img_path.name} -> Question {qnum} not in SELECTED_INDICES")
            continue

        found_selected_images.add(qnum)

        qtext = questions.get(qnum, "").strip()
        if not qtext:
            skipped += 1
            print(f"[SKIP] {img_path.name} -> no question text for Question {qnum}")
            continue

        out_path = OUTPUT_DIR / img_path.name
        try:
            annotate_image_with_footer_text(img_path, qtext, out_path)
            ok += 1
            print(f"[OK]   {img_path.name} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {img_path.name} -> {e}")

    print("\n" + "=" * 60)
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Images dir:     {IMAGES_DIR}")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"Processed:      {total}")
    print(f"Annotated:      {ok}")
    print(f"Skipped:        {skipped}")
    print(f"Skipped (not selected): {skipped_not_selected}")

    if SELECTED_INDICES is None:
        print("Selection mode: all indices")
    else:
        missing_selected = sorted(set(SELECTED_INDICES) - found_selected_images)
        print(f"Selection mode: only {sorted(SELECTED_INDICES)}")
        if missing_selected:
            print(f"Selected indices with no matching image found: {missing_selected}")

    print("=" * 60)


if __name__ == "__main__":
    main()