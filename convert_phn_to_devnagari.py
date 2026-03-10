import os
import glob

DATA_FOLDER = "data"

PHN_TO_DEV = {

    # vowels
    "iy": "ई",
    "ih": "इ",
    "eh": "ए",
    "ae": "ऐ",
    "aa": "आ",
    "ah": "अ",
    "ao": "ऑ",
    "ax": "अ",
    "axr": "अर",
    "er": "अर",
    "ux": "उ",
    "uw": "ऊ",
    "ix": "इ",
    "ay": "आई",
    "ey": "ए",
    "oy": "ओय",
    "aw": "आउ",
    "ow": "ओ",

    # stops
    "p": "प",
    "b": "ब",
    "t": "त",
    "d": "द",
    "k": "क",
    "g": "ग",

    # closures (silent)
    "pcl": "",
    "bcl": "",
    "tcl": "",
    "dcl": "",
    "kcl": "",
    "gcl": "",

    # affricates
    "ch": "च",
    "jh": "झ",

    # fricatives
    "s": "स",
    "sh": "श",
    "z": "ज़",
    "zh": "झ",
    "f": "फ",
    "v": "व",
    "th": "थ",
    "dh": "ध",

    # nasals
    "m": "म",
    "n": "न",
    "ng": "ङ",
    "nx": "ङ",
    "em": "म",
    "en": "न",
    "eng": "ङ",

    # liquids
    "l": "ल",
    "r": "र",
    "el": "ल",

    # semivowels
    "y": "य",
    "w": "व",

    # others
    "h": "ह",
    "hh": "ह",
    "hv": "ह",

    # flap
    "dx": "ड़",

    # silence
    "q": "",
    "epi": "",
    "pau": "",
    "h#": ""
}


def convert_phn_file(filepath):

    with open(filepath, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 3:
            continue

        start, end, phoneme = parts
        phoneme = phoneme.lower()

        dev = PHN_TO_DEV.get(phoneme, phoneme)

        # skip silence phones
        if dev == "":
            continue

        new_lines.append(f"{start} {end} {dev}")

    # overwrite same file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def main():

    phn_files = glob.glob(os.path.join(DATA_FOLDER, "**", "*.PHN"), recursive=True)

    print(f"Found {len(phn_files)} PHN files")

    for phn in phn_files:
        convert_phn_file(phn)
        print("Converted:", phn)

    print("Done.")


if __name__ == "__main__":
    main()
    main()