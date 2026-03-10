from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
import glob
import soundfile as sf

app = Flask(__name__)

# Configuration
DATA_FOLDER = 'data'
ANNOTATIONS_FILE = 'annotations.json'

os.makedirs(DATA_FOLDER, exist_ok=True)

ANNOTATIONS_FOLDER = "annotations"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)


# ==============================
# Annotation Helpers
# ==============================

def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_annotation(data):

    annotations = load_annotations()

    # remove existing annotation for same file + region
    annotations = [
        a for a in annotations
        if not (
            a.get("filename") == data["filename"] and
            a.get("start") == data["start"] and
            a.get("end") == data["end"]
        )
    ]

    # add id and timestamp
    data['id'] = f"{data['filename']}_{data['start']}_{datetime.now().timestamp()}"
    data['timestamp'] = datetime.now().isoformat()

    annotations.append(data)

    with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    return data


def delete_annotation(annotation_id):

    annotations = load_annotations()

    annotations = [a for a in annotations if a.get('id') != annotation_id]

    with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    return True


# ==============================
# Routes
# ==============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/files')
def list_files():

    wav_files = glob.glob(os.path.join(DATA_FOLDER, '*.wav'))
    files = [os.path.basename(f) for f in wav_files]

    return jsonify(files)


@app.route('/api/info/<filename>')
def get_file_info(filename):

    filepath = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    info = sf.info(filepath)

    duration = info.duration
    sr = info.samplerate
    samples = info.frames

    # Load sentence from JSON
    base = os.path.splitext(filename)[0]
    json_file = os.path.join(DATA_FOLDER, base + ".json")

    sentence = ""

    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sentence = data.get("full_sequence", "")
        except:
            pass

    return jsonify({
        "filename": filename,
        "duration": duration,
        "sample_rate": sr,
        "samples": samples,
        "text": sentence
    })

@app.route('/audio/<filename>')
def serve_audio(filename):

    return send_from_directory(DATA_FOLDER, filename)


# ==============================
# PHN loader
# ==============================

@app.route('/api/phn/<filename>')
def get_phn(filename):

    base = os.path.splitext(filename)[0]

    phn_file = os.path.join(DATA_FOLDER, base + ".PHN")

    if not os.path.exists(phn_file):
        return jsonify([])

    wav_path = os.path.join(DATA_FOLDER, filename)

    info = sf.info(wav_path)

    sr = info.samplerate

    phn_data = []

    with open(phn_file, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            start_sample = int(parts[0])
            end_sample = int(parts[1])
            label = parts[2]

            phn_data.append({
                "start": start_sample / sr,
                "end": end_sample / sr,
                "label": label
            })

    return jsonify(phn_data)



@app.route('/api/labels/<filename>')
def get_labels(filename):

    base = os.path.splitext(filename)[0]
    json_file = os.path.join(DATA_FOLDER, base + ".json")

    if not os.path.exists(json_file):
        return jsonify({"frames": [], "sentence": ""})

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return jsonify({
            "sentence": data.get("sentence", ""),
            "frames": data.get("frames", [])
        })

    except Exception as e:
        print("JSON read error:", e)
        return jsonify({"frames": [], "sentence": ""})

# ==============================
# Annotation API
# ==============================

@app.route('/annotate', methods=['POST'])
def annotate():

    data = request.json

    print("Received annotation:", data)

    required = ['filename', 'start', 'end', 'label']

    if not all(field in data for field in required):
        return jsonify({'error': 'Missing required fields'}), 400

    saved = save_annotation(data)

    return jsonify({
        'message': 'Annotation saved',
        'annotation': saved
    })


WINDOW = 0.216

@app.route('/annotations', methods=['GET', 'POST'])
def annotations():

    if request.method == 'GET':
        return jsonify(load_annotations())

    data = request.json

    if not data:
        return jsonify({'error': 'No data'}), 400

    filename = data.get("filename")
    boxes = data.get("boxes")
    label = data.get("label", "")

    if not filename or boxes is None:
        return jsonify({'error': 'Missing filename or boxes'}), 400

    # compute start/end from box index
    start = boxes[0] * WINDOW
    end = (boxes[-1] + 1) * WINDOW

    annotation = {
        "filename": filename,
        "boxes": boxes,
        "start": start,
        "end": end,
        "label": label,
        "auto_saved": True
    }

    saved = save_annotation(annotation)

    return jsonify({
        "message": "Annotation saved",
        "annotation": saved
    })


@app.route('/annotations/<filename>', methods=['GET'])
def get_file_annotations(filename):

    all_ann = load_annotations()

    file_ann = [a for a in all_ann if a.get('filename') == filename]

    return jsonify(file_ann)


@app.route('/submit', methods=['POST'])
def submit_annotation_payload():

    data = request.json

    if not data:
        return jsonify({"error": "No data received"}), 400

    audio_file = data.get("audio_file")

    if not audio_file:
        return jsonify({"error": "audio_file missing"}), 400

    base = os.path.splitext(audio_file)[0]

    output_file = os.path.join(
        ANNOTATIONS_FOLDER,
        base + ".json"
    )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return jsonify({
            "message": "Annotation saved",
            "file": output_file
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/annotations/<annotation_id>', methods=['DELETE'])
def delete_annotation_route(annotation_id):

    success = delete_annotation(annotation_id)

    if success:
        return jsonify({'message': 'Annotation deleted'})

    return jsonify({'error': 'Annotation not found'}), 404


# ==============================
# Start Server
# ==============================

if __name__ == '__main__':

    print("=" * 50)
    print("🚀 Starting WAV Annotation Server")
    print("=" * 50)

    print(f"Data folder: {os.path.abspath(DATA_FOLDER)}")
    print(f"Annotations file: {ANNOTATIONS_FILE}")

    print("\nAvailable WAV files:")

    wavs = glob.glob(os.path.join(DATA_FOLDER, '*.wav'))

    for w in wavs:

        txt = w.replace('.wav', '.txt')

        txt_status = "✓ has text" if os.path.exists(txt) else "✗ no text"

        print(f"  • {os.path.basename(w)} - {txt_status}")

    print("\n" + "=" * 50)

    app.run(debug=True, port=5001, host='0.0.0.0')