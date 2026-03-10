import os
import re

DATA_DIR = "data"

def extract_devanagari_from_phn(phn_path):
    chars = []

    with open(phn_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 3:
                continue

            label = parts[2]

            # keep only devnagari labels
            if re.search(r'[\u0900-\u097F]', label):
                chars.append(label)

    # join with spaces instead of merging
    return " ".join(chars)


def process_txt(txt_path, dev_text):

    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.read().strip()

    # remove leading timestamps
    english_part = re.sub(r'^\d+\s+\d+\s+', '', line)

    new_line = f"{dev_text} {english_part}"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(new_line + "\n")


def main():

    for file in os.listdir(DATA_DIR):

        if not file.lower().endswith(".phn"):
            continue

        base = os.path.splitext(file)[0]

        phn_path = os.path.join(DATA_DIR, file)
        txt_path = os.path.join(DATA_DIR, base + ".txt")

        if not os.path.exists(txt_path):
            continue

        dev_text = extract_devanagari_from_phn(phn_path)

        if not dev_text:
            continue

        process_txt(txt_path, dev_text)

        print(f"Updated: {base}.txt")


if __name__ == "__main__":
    main()