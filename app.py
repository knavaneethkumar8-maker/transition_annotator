from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, session
import os
import json
from datetime import datetime, timedelta, timezone
import glob
import soundfile as sf
import hashlib
import uuid
import pytz

app = Flask(__name__)
app.secret_key = "annotator_secret_key_2024"  # Changed for security

# Folder structure
DATA_FOLDER = 'data'
ANNOTATIONS_FOLDER = "annotations"
USERS_FILE = "users.json"
FILE_STATUS_FILE = "file_status.json"
# Track skipped files per user (in memory)
user_skips = {}

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)

# ==============================
# AKSHAR TRACKING - SIMPLIFIED
# ==============================
AKSHAR_TRACKING_FILE = "akshar_tracking.json"
AKSHAR_DAILY_TARGET = 1000
AKSHAR_OVERALL_TARGET = 5000

# Indian timezone
IST = timezone(timedelta(hours=5, minutes=30))

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def get_current_ist_date():
    """Get current date in IST (YYYY-MM-DD)"""
    return get_ist_now().strftime("%Y-%m-%d")

def load_akshar_tracking():
    """Load akshar tracking data from file"""
    if os.path.exists(AKSHAR_TRACKING_FILE):
        try:
            with open(AKSHAR_TRACKING_FILE, 'r', encoding='utf-8') as f:
                tracking = json.load(f)
                
            # Check if we need to reset daily counts (new day in IST)
            current_date = get_current_ist_date()
            if tracking.get("last_reset") != current_date:
                # Reset daily counts for new day
                tracking["daily"] = {}
                tracking["last_reset"] = current_date
                save_akshar_tracking(tracking)
            
            return tracking
        except:
            return init_akshar_tracking()
    else:
        return init_akshar_tracking()

def init_akshar_tracking():
    """Initialize akshar tracking structure"""
    tracking = {
        "daily": {},  # Format: {date: {username: count}}
        "overall": {},  # Format: {username: total_count}
        "last_reset": get_current_ist_date()
    }
    save_akshar_tracking(tracking)
    return tracking

def save_akshar_tracking(tracking):
    """Save akshar tracking to file"""
    with open(AKSHAR_TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, indent=2)

def update_akshar_counts(username, frames):
    """Update akshar counts for a user based on submitted frames"""
    # Count non-empty frames (akshars)
    akshar_count = sum(1 for frame in frames if frame.get("text") and frame["text"].strip() != "")
    
    if akshar_count == 0:
        return {"added": 0}
    
    tracking = load_akshar_tracking()
    current_date = get_current_ist_date()
    
    # Initialize daily structure for current date if needed
    if current_date not in tracking["daily"]:
        tracking["daily"][current_date] = {}
    
    # Update daily count
    tracking["daily"][current_date][username] = tracking["daily"][current_date].get(username, 0) + akshar_count
    
    # Update overall count
    tracking["overall"][username] = tracking["overall"].get(username, 0) + akshar_count
    
    # Save updated tracking
    save_akshar_tracking(tracking)
    
    return {
        "added": akshar_count,
        "user_daily": tracking["daily"][current_date][username],
        "user_overall": tracking["overall"][username]
    }

def get_akshar_stats(username):
    """Get simple akshar statistics"""
    tracking = load_akshar_tracking()
    current_date = get_current_ist_date()
    
    # Get daily stats
    daily_stats = tracking["daily"].get(current_date, {})
    total_daily = sum(daily_stats.values())
    
    # Get user stats
    user_daily = daily_stats.get(username, 0)
    user_overall = tracking["overall"].get(username, 0)
    total_overall = sum(tracking["overall"].values())
    
    return {
        "date": current_date,
        "user_daily": user_daily,
        "user_overall": user_overall,
        "total_daily": total_daily,
        "total_overall": total_overall,
        "daily_target": AKSHAR_DAILY_TARGET,
        "overall_target": AKSHAR_OVERALL_TARGET
    }

# ==============================
# FILE STATUS MANAGEMENT
# ==============================

def init_file_status():
    """Initialize file status tracking if it doesn't exist"""
    if not os.path.exists(FILE_STATUS_FILE):

        # ONLY include _4x files for annotation
        wav_files = glob.glob(os.path.join(DATA_FOLDER, '*_4x.wav'))

        file_status = {}

        for wav_file in wav_files:
            filename = os.path.basename(wav_file)

            file_status[filename] = {
                "status": "pending",  # pending, assigned, completed
                "assigned_to": None,
                "assigned_at": None,
                "completed_at": None,
                "annotation_file": None
            }

        save_file_status(file_status)
        return file_status

    with open(FILE_STATUS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_file_status(file_status):
    """Save file status to JSON"""
    with open(FILE_STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(file_status, f, indent=2, ensure_ascii=False)

def get_user_annotation_dir(username):
    """Get or create user's annotation directory"""
    user_dir = os.path.join(ANNOTATIONS_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_user_completed_files(username):
    """Get list of files completed by a user"""
    user_dir = get_user_annotation_dir(username)
    completed_files = []
    
    if os.path.exists(user_dir):
        for json_file in glob.glob(os.path.join(user_dir, "*.json")):
            filename = os.path.basename(json_file)
            # Extract original filename (remove _completed or just the basename)
            completed_files.append(filename.replace('_completed.json', '.wav'))
    
    return completed_files

def get_next_file_for_user(username):
    """Get next unassigned or pending file for user"""
    file_status = init_file_status()
    user_completed = get_user_completed_files(username)

    skipped_files = user_skips.get(username, [])

    # If user already has assigned file
    for filename, status in file_status.items():
        if status["assigned_to"] == username and status["status"] == "assigned":
            if filename not in user_completed:
                return filename

    # Pass 1: pending files NOT skipped
    for filename, status in file_status.items():
        if (
            status["status"] == "pending"
            and filename not in user_completed
            and filename not in skipped_files
        ):
            status["status"] = "assigned"
            status["assigned_to"] = username
            status["assigned_at"] = datetime.now().isoformat()
            save_file_status(file_status)
            return filename

    # Pass 2: if all files were skipped, reset skip list and allow them again
    if username in user_skips:
        user_skips[username] = []

    for filename, status in file_status.items():
        if status["status"] == "pending" and filename not in user_completed:
            status["status"] = "assigned"
            status["assigned_to"] = username
            status["assigned_at"] = datetime.now().isoformat()
            save_file_status(file_status)
            return filename
        
    return None

def mark_file_completed(filename, username, annotation_filename):
    """Mark a file as completed by user"""
    file_status = init_file_status()
    
    if filename in file_status:
        file_status[filename]["status"] = "completed"
        file_status[filename]["completed_at"] = datetime.now().isoformat()
        file_status[filename]["annotation_file"] = annotation_filename
        save_file_status(file_status)
        return True
    
    return False

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
        return jsonify({"error": "User exists"}), 400

    users.append({
        "username": username,
        "password": hash_password(password)
    })

    save_users(users)
    
    # Create user annotation directory
    get_user_annotation_dir(username)

    return jsonify({"message": "registered"})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    users = load_users()

    for u in users:
        if u["username"] == username and u["password"] == hash_password(password):
            session["user"] = username
            return jsonify({"success": True})

    return jsonify({"error": "invalid login"}), 401

@app.route("/logout")
def logout():
    username = session.get("user")
    if username in user_skips:
        del user_skips[username]

    session.clear()
    return redirect("/login")

def require_login():
    return "user" in session

# ==============================
# Find matching normal WAV file
# ==============================
@app.route('/api/matching-wav/<filename>')
def get_matching_wav(filename):
    """Find a matching normal WAV file for comparison"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    # Remove _4x from filename to get normal version
    normal_filename = filename.replace('_4x', '')
    
    # Check if normal version exists
    normal_path = os.path.join(DATA_FOLDER, normal_filename)
    if os.path.exists(normal_path):
        return jsonify({
            "filename": normal_filename,
            "found": True
        })
    
    return jsonify({
        "filename": None,
        "found": False
    })
    
@app.route('/matching-audio/<filename>')
def serve_matching_audio(filename):
    """Serve matching audio file"""
    if not require_login():
        return redirect("/login")
    return send_from_directory(DATA_FOLDER, filename)

# ==============================
# ROUTES
# ==============================

@app.route("/")
def index():
    if not require_login():
        return redirect("/login")
    return render_template("index.html", user=session["user"])

@app.route('/api/next-file')
def get_next_file():
    """Get the next file to annotate for the current user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    next_file = get_next_file_for_user(username)
    
    if next_file:
        return jsonify({
            "filename": next_file,
            "message": "File assigned successfully"
        })
    else:
        return jsonify({
            "filename": None,
            "message": "No more files to annotate"
        })

@app.route('/api/current-file')
def get_current_file():
    """Get the currently assigned file for user (if any)"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    file_status = init_file_status()
    
    # Check if user has any assigned file
    for filename, status in file_status.items():
        if status["assigned_to"] == username and status["status"] == "assigned":
            return jsonify({"filename": filename})
    
    return jsonify({"filename": None})

@app.route('/api/file-info/<filename>')
def get_file_info(filename):
    """Get information about a specific file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    filepath = os.path.join(DATA_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "file not found"}), 404
    
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
            sentence = data.get("sentence", "")
    
    return jsonify({
        "filename": filename,
        "duration": duration,
        "sample_rate": sr,
        "samples": samples,
        "text": sentence
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio file"""
    if not require_login():
        return redirect("/login")
    return send_from_directory(DATA_FOLDER, filename)

@app.route('/api/phn/<filename>')
def get_phn(filename):
    """Get PHN annotations for a file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
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
    """Get pre-existing labels for a file"""
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

@app.route('/api/user-progress')
def get_user_progress():
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    file_status = init_file_status()
    
    total_files = len(file_status)

    # Global completed
    global_completed_count = sum(
        1 for f in file_status.values()
        if f["status"] == "completed"
    )

    # User completed
    user_completed_count = sum(
        1 for f in file_status.values()
        if f["status"] == "completed"
        and f["assigned_to"] == username
    )

    # Current file
    current_file = None
    for filename, status in file_status.items():
        if status["assigned_to"] == username and status["status"] == "assigned":
            current_file = filename
            break

    return jsonify({
        "username": username,
        "total_files": total_files,
        "completed": user_completed_count,
        "global_completed": global_completed_count,
        "remaining": total_files - global_completed_count,
        "current_file": current_file
    })

@app.route('/api/akshar-stats')
def akshar_stats():
    """Get akshar statistics for the current user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    stats = get_akshar_stats(username)
    return jsonify(stats)

@app.route('/submit', methods=['POST'])
def submit_annotation():
    """Submit annotation for a file"""
    if not require_login():
        return jsonify({"error": "login required"}), 401
    
    data = request.json
    audio_file = data.get("audio_file")
    username = session["user"]
    frames = data.get("frames", [])
    
    # Update akshar counts
    akshar_update = update_akshar_counts(username, frames)
    
    # Generate unique filename for annotation
    base = os.path.splitext(audio_file)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotation_filename = f"{base}_{timestamp}.json"
    
    # Save to user's directory
    user_dir = get_user_annotation_dir(username)
    output_file = os.path.join(user_dir, annotation_filename)
    
    # Construct output data
    output_data = {
        "audio_file": audio_file,
        "annotator": username,
        "timestamp": datetime.now().isoformat(),
        "window_ms": data.get("window_ms"),
        "sentence": data.get("sentence"),
        "full_sequence": data.get("full_sequence"),
        "frames": frames
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Mark file as completed
    mark_file_completed(audio_file, username, annotation_filename)
    
    # Include akshar stats in response
    response_data = {
        "message": "saved",
        "file": annotation_filename,
        "next_file": get_next_file_for_user(username),
        "akshar": akshar_update
    }
    
    return jsonify(response_data)

@app.route('/api/skip-file', methods=['POST'])
def skip_file():
    """Skip current file and get next one"""
    if not require_login():
        return jsonify({"error": "login required"}), 401

    username = session["user"]
    data = request.json
    current_file = data.get("current_file")

    file_status = init_file_status()

    # Track skips per user
    if username not in user_skips:
        user_skips[username] = []

    if current_file and current_file not in user_skips[username]:
        user_skips[username].append(current_file)

    # Release the file
    if current_file in file_status:
        file_status[current_file]["status"] = "pending"
        file_status[current_file]["assigned_to"] = None
        file_status[current_file]["assigned_at"] = None

    save_file_status(file_status)

    next_file = get_next_file_for_user(username)

    return jsonify({
        "message": "file skipped",
        "next_file": next_file
    })

# Initialize file status on startup
init_file_status()
# Initialize akshar tracking on startup
load_akshar_tracking()

if __name__ == '__main__':
    print("=" * 50)
    print("WAV Annotation Server - Distributed Mode")
    print("=" * 50)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Annotations folder: {ANNOTATIONS_FOLDER}")
    print(f"Users: {USERS_FILE}")
    print(f"File status: {FILE_STATUS_FILE}")
    print(f"Akshar tracking: {AKSHAR_TRACKING_FILE}")
    print(f"Daily target: {AKSHAR_DAILY_TARGET} akshars")
    print(f"Overall target: {AKSHAR_OVERALL_TARGET} akshars")
    print("=" * 50)
    
    app.run(debug=True, port=5001, host='0.0.0.0')