import os
import json

DATA_DIR = "data"

def process_file(base):
    s_file = os.path.join(DATA_DIR, f"{base}_s.txt")
    json_file = os.path.join(DATA_DIR, f"{base}_4x.json")

    if not os.path.exists(s_file) or not os.path.exists(json_file):
        return

    # read sequence
    with open(s_file, "r", encoding="utf-8") as f:
        seq = [x.strip() for x in f.read().strip().split(",") if x.strip()]

    # read json
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]

    seq_index = 0

    for frame in frames:
        if frame["text"] != "":
            if seq_index < len(seq):
                frame["text"] = seq[seq_index]
                seq_index += 1
            else:
                break

    # update full sequence
    data["full_sequence"] = ",".join(seq)

    # save
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated {base}")

def main():
    for file in os.listdir(DATA_DIR):
        if file.endswith("_s.txt"):
            base = file.replace("_s.txt", "")
            process_file(base)

if __name__ == "__main__":
    main()
