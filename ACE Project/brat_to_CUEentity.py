import os
import re
from pathlib import Path
import json


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

            # --- ONLY CUE ENTITIES
            if not label.endswith("_CUE"):
                continue

            start = int(tag_info[1])
            end = int(tag_info[-1]) #if len(tag_info[1:]) == 2 else int(tag_info[-1])

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


def assign_cue_labels(tokens, entities):
    cue_types = sorted({
        ent["label"]
        for ent in entities
    })

    labels = {
        cue: ["O"] * len(tokens)
        for cue in cue_types
    }

    for ent in entities:
        ent_start = ent["start"]
        ent_end = ent["end"]
        cue = ent["label"]

        for i, (_, start, end) in enumerate(tokens):

            if end <= ent_start or start >= ent_end:
                continue

            labels[cue][i] = cue

    return labels

def brat_to_dict(txt_path, ann_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    entities = parse_ann_file(ann_path)
    tokens = tokenize_with_offsets(text)
    labels = assign_cue_labels(tokens, entities)

    return {
        "tokens": [tok for tok, _, _ in tokens],
        **labels
    }


def convert_folder(brat_dir, output_dir):
    brat_dir = Path(brat_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_files = []

    for txt_file in sorted(brat_dir.glob("*.txt")):
        ann_file = txt_file.with_suffix(".ann")

        if not ann_file.exists():
            continue

        try:
            data = brat_to_dict(txt_file, ann_file)
            if not data:
                continue

            out_file = output_dir / f"{txt_file.stem}.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            failed_files.append((txt_file.name, str(e)))

    print("\n" + "=" * 80)
    print("CONVERSION FINISHED")
    print("=" * 80)

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):\n")

        for fname, error in failed_files:
            print(f"{fname}")
            print(f"  -> {error}")

    else:
        print("\nNo errors found.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    input_dir = BASE_DIR / "brat_annotated_v1"
    input_dir = Path("C:/Users/ASUS/Berlin/Practicas/ACE/brat/data/brat_annotated_v1_corrected_BA")

    convert_folder(input_dir, BASE_DIR / "CUE_annotated_v1_correctedBA")