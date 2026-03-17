import os
import json

DATA_DIR = "./data"   # change if your path is different

for file in os.listdir(DATA_DIR):
    if file.endswith(".json"):
        path = os.path.join(DATA_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # go through frames
        if "frames" in data:
            for frame in data["frames"]:
                if frame.get("text") in ["shunya", "vyanjan"]:
                    frame["text"] = ""

        # save back
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

print("Done cleaning JSON files.")
