from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, session
import os
import json
from datetime import datetime
import glob
import soundfile as sf
import hashlib

app = Flask(__name__)
app.secret_key = "annotator_secret_key"

DATA_FOLDER = 'data'
ANNOTATIONS_FILE = 'annotations.json'
ANNOTATIONS_FOLDER = "annotations"
USERS_FILE = "users.json"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)


# ==============================
# USER SYSTEM
# ==============================

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/api/register", methods=["POST"])
def register():

    data = request.json
    username = data.get("username")
    password = data.get("password")

    users = load_users()

    if any(u["username"] == username for u in users):
        return jsonify({"error":"User exists"}),400

    users.append({
        "username": username,
        "password": hash_password(password)
    })

    save_users(users)

    return jsonify({"message":"registered"})


@app.route("/api/login", methods=["POST"])
def login():

    data = request.json
    username = data.get("username")
    password = data.get("password")

    users = load_users()

    for u in users:
        if u["username"] == username and u["password"] == hash_password(password):
            session["user"] = username
            return jsonify({"success":True})

    return jsonify({"error":"invalid login"}),401


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


def require_login():
    if "user" not in session:
        return False
    return True


# ==============================
# ROUTES
# ==============================

@app.route("/")
def index():

    if not require_login():
        return redirect("/login")

    return render_template("index.html", user=session["user"])


@app.route('/api/files')
def list_files():

    if not require_login():
        return jsonify({"error":"not logged in"}),401

    wav_files = glob.glob(os.path.join(DATA_FOLDER, '*.wav'))
    files = [os.path.basename(f) for f in wav_files]

    return jsonify(files)


@app.route('/api/info/<filename>')
def get_file_info(filename):

    filepath = os.path.join(DATA_FOLDER, filename)

    info = sf.info(filepath)

    duration = info.duration
    sr = info.samplerate
    samples = info.frames

    base = os.path.splitext(filename)[0]
    json_file = os.path.join(DATA_FOLDER, base + ".json")

    sentence = ""

    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            sentence = data.get("full_sequence","")

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


@app.route('/annotations', methods=['GET', 'POST'])
def get_annotations():

    if not require_login():
        return jsonify({"error": "login required"}), 401

    if not os.path.exists(ANNOTATIONS_FILE):
        return jsonify([])

    try:
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return jsonify(data)
    except:
        return jsonify([])


@app.route('/api/labels/<filename>')
def get_labels(filename):

    if not require_login():
        return jsonify({"error": "login required"}), 401

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


@app.route('/annotate', methods=['POST'])
def annotate():

    if not require_login():
        return jsonify({"error": "login required"}), 401

    data = request.json

    if not os.path.exists(ANNOTATIONS_FILE):
        annotations = []
    else:
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            annotations = json.load(f)

    data["annotator"] = session["user"]
    data["timestamp"] = datetime.now().isoformat()

    annotations.append(data)

    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    return jsonify({"message": "saved"})
# ==============================
# ANNOTATION SAVE
# ==============================

WINDOW = 0.216

@app.route('/submit', methods=['POST'])
def submit_annotation_payload():

    if not require_login():
        return jsonify({"error": "login required"}), 401

    data = request.json

    audio_file = data.get("audio_file")
    user = session["user"]

    base = os.path.splitext(audio_file)[0]

    output_file = os.path.join(
        ANNOTATIONS_FOLDER,
        f"{base}.json"
    )

    # construct ordered output
    output_data = {
        "audio_file": data.get("audio_file"),
        "annotator": user,
        "timestamp": datetime.now().isoformat(),
        "window_ms": data.get("window_ms"),
        "sentence": data.get("sentence"),
        "full_sequence": data.get("full_sequence"),
        "frames": data.get("frames")
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return jsonify({
        "message": "saved",
        "file": output_file
    })
# ==============================
# Start Server
# ==============================

if __name__ == '__main__':

    print("=" * 50)
    print("WAV Annotation Server")
    print("=" * 50)

    app.run(debug=True, port=5001, host='0.0.0.0')