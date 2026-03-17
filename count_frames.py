import os
import json
import glob

ANNOTATIONS_DIR = "annotations"

total_frames = 0
akshar_frames = 0
empty_frames = 0

files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*", "*.json"))

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])

    for frame in frames:
        total_frames += 1

        text = frame.get("text")

        if text is None:
            empty_frames += 1
            continue

        text = str(text).strip()

        if text == "":
            empty_frames += 1
        else:
            akshar_frames += 1

print("="*40)
print("FILES SCANNED:", len(files))
print("="*40)
print("TOTAL FRAMES:", total_frames)
print("AKSHAR FRAMES:", akshar_frames)
print("EMPTY / NO-AKSHAR FRAMES:", empty_frames)
print("="*40)
