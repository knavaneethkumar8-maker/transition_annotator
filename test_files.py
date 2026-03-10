import os

DATA_FOLDER = 'data'

print("Checking files in data folder:")
print("=" * 40)

for f in os.listdir(DATA_FOLDER):
    if f.endswith('.wav'):
        wav_path = os.path.join(DATA_FOLDER, f)
        txt_path = wav_path.replace('.wav', '.txt')
        
        print(f"\nFile: {f}")
        print(f"  WAV exists: {os.path.exists(wav_path)}")
        print(f"  TXT exists: {os.path.exists(txt_path)}")
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as txt:
                content = txt.read().strip()
                print(f"  Content: '{content}'")
