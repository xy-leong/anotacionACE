import os
import re
from pathlib import Path


def parse_ann_file(ann_path):
    entities = []

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("T"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            tag_info = parts[1].split()
            label = tag_info[0]

            # --- ONLY ACE ENTITIES
            if not label.startswith("ACE_"):
                continue

            start = int(tag_info[1])
            end = int(tag_info[2])

            entities.append({
                "start": start,
                "end": end,
                "label": label
            })

    return entities


def tokenize_with_offsets(text):
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def assign_bio_labels(tokens, entities):
    bio_labels = ["O"] * len(tokens)

    for ent in entities:
        ent_start = ent["start"]
        ent_end = ent["end"]
        label = ent["label"]

        first_token = True

        for i, (_, start, end) in enumerate(tokens):

            # Skip non-overlapping tokens
            if end <= ent_start or start >= ent_end:
                continue

            if first_token:
                bio_labels[i] = f"B-{label}"
                first_token = False
            else:
                bio_labels[i] = f"I-{label}"

    return bio_labels


def brat_to_bio(txt_path, ann_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    entities = parse_ann_file(ann_path)
    tokens = tokenize_with_offsets(text)
    labels = assign_bio_labels(tokens, entities)

    return [(tok, lab) for (tok, _, _), lab in zip(tokens, labels)]


def convert_folder(brat_dir, output_dir):
    brat_dir = Path(brat_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(brat_dir.glob("*.txt")):
        ann_file = txt_file.with_suffix(".ann")

        if not ann_file.exists():
            continue

        bio_data = brat_to_bio(txt_file, ann_file)
        if not bio_data:
            continue

        out_file = output_dir / f"{txt_file.stem}.bio"

        with open(out_file, "w", encoding="utf-8", newline="\n") as f:
            for token, label in bio_data:
                f.write(f"{token}\t{label}\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    convert_folder(BASE_DIR / "brat_output", BASE_DIR / "BIO_ACE")