import os
import subprocess

INPUT_DIR = "input_data"
OUTPUT_DIR = "data"

SLOW_FACTOR = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


def slow_wav(input_wav, output_wav):
    """
    Slow audio 4x using sox
    """
    cmd = [
        "sox",
        input_wav,
        output_wav,
        "tempo",
        str(1.0 / SLOW_FACTOR)
    ]
    subprocess.run(cmd, check=True)


def convert_phn(input_phn, output_phn):

    with open(input_phn, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 3:
            continue

        start = int(parts[0]) * SLOW_FACTOR
        end = int(parts[1]) * SLOW_FACTOR
        label = parts[2]

        new_lines.append(f"{start} {end} {label}")

    with open(output_phn, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def process():

    files = os.listdir(INPUT_DIR)

    wav_files = [f for f in files if f.endswith(".wav")]

    for wav in wav_files:

        base = os.path.splitext(wav)[0]

        wav_path = os.path.join(INPUT_DIR, wav)
        txt_path = os.path.join(INPUT_DIR, base + ".txt")
        phn_path = os.path.join(INPUT_DIR, base + ".PHN")

        if not os.path.exists(txt_path) or not os.path.exists(phn_path):
            print(f"Skipping {base}, missing txt or PHN")
            continue

        out_wav = os.path.join(OUTPUT_DIR, base + "_4x.wav")
        out_txt = os.path.join(OUTPUT_DIR, base + "_4x.txt")
        out_phn = os.path.join(OUTPUT_DIR, base + "_4x.PHN")

        print("Processing:", base)

        # slow audio
        slow_wav(wav_path, out_wav)

        # copy text
        with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.read()

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(txt)

        # convert phn timestamps
        convert_phn(phn_path, out_phn)


if __name__ == "__main__":
    process()