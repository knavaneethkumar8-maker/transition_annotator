import os
import random
import shutil

wav_dir = "/data/safe_storage/indic-voice/wav"
txt_dir = "/data/safe_storage/indic-voice/txt"
output_dir = "/root/annotator/annotate_app/INDIC_DATA"

os.makedirs(output_dir, exist_ok=True)

wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

pairs = []
for w in wav_files:
    base = os.path.splitext(w)[0]
    txt_file = os.path.join(txt_dir, base + ".txt")

    if os.path.exists(txt_file):
        pairs.append(base)

print("Total matching pairs:", len(pairs))

sample_size = min(100, len(pairs))
selected = random.sample(pairs, sample_size)

for base in selected:
    folder = os.path.join(output_dir, base)
    os.makedirs(folder, exist_ok=True)

    shutil.copy(os.path.join(wav_dir, base + ".wav"), folder)
    shutil.copy(os.path.join(txt_dir, base + ".txt"), folder)

print("Copied", sample_size, "pairs")
