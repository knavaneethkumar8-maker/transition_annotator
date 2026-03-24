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
AUTOSAVE_FOLDER = "autosave"  # New folder for auto-saves
USERS_FILE = "users.json"
FILE_STATUS_FILE = "file_status.json"
# Track skipped files per user (in memory)
user_skips = {}

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(AUTOSAVE_FOLDER, exist_ok=True)  # Create autosave folder

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
    if os.path.exists(AKSHAR_TRACKING_FILE):
        try:
            with open(AKSHAR_TRACKING_FILE, 'r', encoding='utf-8') as f:
                tracking = json.load(f)
        except Exception as e:
            print("Error reading akshar file:", e)
            return init_akshar_tracking()

        current_date = get_current_ist_date()

        # ✅ ONLY reset daily, NOT overall
        if tracking.get("last_reset") != current_date:
            tracking["daily"] = {}  # reset daily
            tracking["last_reset"] = current_date

            # ⚠️ DO NOT TOUCH overall
            save_akshar_tracking(tracking)

        return tracking

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

# Add these new variables near the top with other tracking constants
DURATION_TRACKING_FILE = "duration_tracking.json"

def load_duration_tracking():
    """Load duration tracking data"""
    if os.path.exists(DURATION_TRACKING_FILE):
        try:
            with open(DURATION_TRACKING_FILE, 'r', encoding='utf-8') as f:
                tracking = json.load(f)
        except Exception as e:
            print("Error reading duration file:", e)
            return init_duration_tracking()

        current_date = get_current_ist_date()

        # Reset daily if date changed
        if tracking.get("last_reset") != current_date:
            tracking["daily"] = {}  # reset daily
            tracking["last_reset"] = current_date
            save_duration_tracking(tracking)

        return tracking
    else:
        return init_duration_tracking()

def init_duration_tracking():
    """Initialize duration tracking structure"""
    tracking = {
        "daily": {},  # Format: {date: {username: total_seconds}}
        "overall": {},  # Format: {username: total_seconds}
        "last_reset": get_current_ist_date()
    }
    save_duration_tracking(tracking)
    return tracking

def save_duration_tracking(tracking):
    """Save duration tracking to file"""
    with open(DURATION_TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, indent=2)

def update_duration_counts(username, duration_seconds):
    """Update duration counts for a user based on submitted file"""
    if duration_seconds <= 0:
        return {"added": 0}
    
    tracking = load_duration_tracking()
    current_date = get_current_ist_date()
    
    # Initialize daily structure for current date if needed
    if current_date not in tracking["daily"]:
        tracking["daily"][current_date] = {}
    
    # Update daily count
    tracking["daily"][current_date][username] = tracking["daily"][current_date].get(username, 0) + duration_seconds
    
    # Update overall count
    tracking["overall"][username] = tracking["overall"].get(username, 0) + duration_seconds
    
    # Save updated tracking
    save_duration_tracking(tracking)
    
    return {
        "added": duration_seconds,
        "user_daily": tracking["daily"][current_date][username],
        "user_overall": tracking["overall"][username]
    }

def get_duration_stats(username):
    """Get duration statistics for user"""
    tracking = load_duration_tracking()
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
        "user_daily": round(user_daily, 1),  # Round to 1 decimal
        "user_overall": round(user_overall, 1),
        "total_daily": round(total_daily, 1),
        "total_overall": round(total_overall, 1)
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
# AUTO-SAVE FUNCTIONALITY
# ==============================

def get_autosave_path(username, filename):
    """Get path for auto-save file"""
    # Create username subfolder in autosave
    user_autosave_dir = os.path.join(AUTOSAVE_FOLDER, username)
    os.makedirs(user_autosave_dir, exist_ok=True)
    
    # Auto-save filename: originalname_autosave.json
    base = os.path.splitext(filename)[0]
    autosave_filename = f"{base}_autosave.json"
    return os.path.join(user_autosave_dir, autosave_filename)

@app.route('/api/autosave', methods=['POST'])
def autosave():
    """Auto-save current progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    username = session["user"]
    audio_file = data.get("audio_file")
    frames = data.get("frames", [])
    
    if not audio_file:
        return jsonify({"error": "no file specified"}), 400
    
    # Save to autosave folder
    autosave_path = get_autosave_path(username, audio_file)
    
    autosave_data = {
        "audio_file": audio_file,
        "annotator": username,
        "last_updated": datetime.now().isoformat(),
        "frames": frames
    }
    
    with open(autosave_path, 'w', encoding='utf-8') as f:
        json.dump(autosave_data, f, indent=2, ensure_ascii=False)
    
    return jsonify({"message": "autosaved", "timestamp": datetime.now().isoformat()})


@app.route('/api/duration-stats')
def duration_stats():
    """Get duration statistics for the current user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    stats = get_duration_stats(username)
    return jsonify(stats)

@app.route('/api/autosave/<filename>', methods=['GET'])
def get_autosave(filename):
    """Get auto-saved progress for a file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        with open(autosave_path, 'r', encoding='utf-8') as f:
            autosave_data = json.load(f)
        return jsonify(autosave_data)
    
    return jsonify({"frames": []})

@app.route('/api/autosave/clear/<filename>', methods=['POST'])
def clear_autosave(filename):
    """Clear auto-saved progress after successful submission"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    
    return jsonify({"message": "autosave cleared"})

# ==============================
# FILE STATUS MANAGEMENT
# ==============================

def init_file_status():
    """Initialize file status tracking if it doesn't exist, preserving existing data"""
    # Get all _4x.wav files (only these should be annotated)
    wav_files = glob.glob(os.path.join(DATA_FOLDER, '*_4x.wav'))
    
    # Load existing file status if it exists
    existing_status = {}
    if os.path.exists(FILE_STATUS_FILE):
        try:
            with open(FILE_STATUS_FILE, 'r', encoding='utf-8') as f:
                existing_status = json.load(f)
        except Exception as e:
            print("Error reading existing file status:", e)
            existing_status = {}
    
    # Check for new files
    new_files = []
    updated_status = existing_status.copy()
    
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        
        # If file doesn't exist in status, add it
        if filename not in updated_status:
            print(f"Adding new file: {filename}")
            new_files.append(filename)
            updated_status[filename] = {
                "status": "pending",  # pending, assigned, completed
                "assigned_to": None,
                "assigned_at": None,
                "completed_at": None,
                "annotation_file": None,
                "priority": 1 if filename.startswith('BEEJ_') else 0  # Add priority flag
            }
        else:
            # Ensure priority flag exists for existing files
            if "priority" not in updated_status[filename]:
                updated_status[filename]["priority"] = 1 if filename.startswith('BEEJ_') else 0
    
    # Only save if there were changes
    if new_files:
        print(f"Added {len(new_files)} new files to tracking")
        save_file_status(updated_status)
    elif updated_status != existing_status:
        # Save if we added priority flags to existing files
        save_file_status(updated_status)
    
    return updated_status

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
    """Get next unassigned or pending file for user, prioritizing BEEJ_ files"""
    file_status = init_file_status()
    user_completed = get_user_completed_files(username)
    skipped_files = user_skips.get(username, [])
    
    # If user already has assigned file
    for filename, status in file_status.items():
        if status.get("assigned_to") == username and status.get("status") == "assigned":
            if filename not in user_completed:
                return filename
    
    # Check if there are any available BEEJ_ files (not assigned to anyone, not completed)
    available_beej_files = []
    for filename, status in file_status.items():
        if (status.get("priority", 0) == 1 and 
            status.get("status") == "pending" and 
            filename not in user_completed and
            filename not in skipped_files):
            available_beej_files.append((filename, status))
    
    # If there are available BEEJ_ files, assign one
    if available_beej_files:
        available_beej_files.sort(key=lambda x: x[0])  # Sort by filename
        filename, status = available_beej_files[0]
        status["status"] = "assigned"
        status["assigned_to"] = username
        status["assigned_at"] = datetime.now().isoformat()
        save_file_status(file_status)
        return filename
    
    # Check if there are any BEEJ_ files at all (in any state)
    total_beej_files = []
    beej_completed = []
    beej_assigned_to_others = []
    
    for filename, status in file_status.items():
        if status.get("priority", 0) == 1:
            total_beej_files.append(filename)
            if filename in user_completed or status.get("status") == "completed":
                beej_completed.append(filename)
            elif status.get("status") == "assigned" and status.get("assigned_to") != username:
                beej_assigned_to_others.append(filename)
    
    # If there are BEEJ_ files that are assigned to others (in progress)
    # but NO available BEEJ_ files for this user, then we can assign regular files
    # This means all BEEJ_ files are either completed OR being worked on by others
    
    # Get pending regular files (not completed, not skipped)
    pending_regular_files = []
    for filename, status in file_status.items():
        if (status.get("priority", 0) != 1 and 
            status.get("status") == "pending" and 
            filename not in user_completed and 
            filename not in skipped_files):
            pending_regular_files.append((filename, status))
    
    # Assign regular files if available
    if pending_regular_files:
        filename, status = pending_regular_files[0]
        status["status"] = "assigned"
        status["assigned_to"] = username
        status["assigned_at"] = datetime.now().isoformat()
        save_file_status(file_status)
        return filename
    
    # If all files were skipped, reset skip list and try again
    if username in user_skips:
        user_skips[username] = []
    
    # Final attempt: any pending file (including those that were skipped)
    for filename, status in file_status.items():
        if status.get("status") == "pending" and filename not in user_completed:
            status["status"] = "assigned"
            status["assigned_to"] = username
            status["assigned_at"] = datetime.now().isoformat()
            save_file_status(file_status)
            return filename
    
    return None

def has_priority_files_left(username):
    """Check if there are any priority (BEEJ_) files left for the user"""
    file_status = init_file_status()
    user_completed = get_user_completed_files(username)
    
    for filename, status in file_status.items():
        if (status.get("priority", 0) == 1 and 
            status.get("status") != "completed" and 
            filename not in user_completed):
            return True
    return False

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
    
    # Get file duration
    filepath = os.path.join(DATA_FOLDER, audio_file)
    duration_seconds = 0
    if os.path.exists(filepath):
        info = sf.info(filepath)
        duration_seconds = info.duration
    
    # Update akshar counts
    akshar_update = update_akshar_counts(username, frames)
    
    # Update duration counts
    duration_update = update_duration_counts(username, duration_seconds)
    
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
    
    # Clear auto-save for this file
    autosave_path = get_autosave_path(username, audio_file)
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    
    # Include both stats in response
    response_data = {
        "message": "saved",
        "file": annotation_filename,
        "next_file": get_next_file_for_user(username),
        "akshar": akshar_update,
        "duration": duration_update  # Add duration data
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
    
    # Clear auto-save for skipped file
    autosave_path = get_autosave_path(username, current_file)
    if os.path.exists(autosave_path):
        os.remove(autosave_path)

    # Check if current file is a BEEJ_ file
    is_beej = current_file.startswith('BEEJ_') if current_file else False
    
    next_file = get_next_file_for_user(username)
    
    # Add a message about BEEJ_ files if relevant
    message = "file skipped"
    if is_beej and not next_file:
        # Check if there are BEEJ_ files assigned to others
        file_status = init_file_status()
        beej_in_progress = False
        for filename, status in file_status.items():
            if (status.get("priority", 0) == 1 and 
                status.get("status") == "assigned" and 
                status.get("assigned_to") != username):
                beej_in_progress = True
                break
        
        if beej_in_progress:
            message = "BEEJ_ files are being processed by others. Please wait."

    return jsonify({
        "message": message,
        "next_file": next_file
    })

# ==============================
# ADMIN STATS PAGE
# ==============================

@app.route('/stats')
def stats_page():
    """Show statistics page for all annotators"""
    if not require_login():
        return redirect("/login")
    return render_template("stats.html", user=session["user"])

@app.route('/api/all-stats')
def get_all_stats():
    """Get statistics for all annotators"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    # Load akshar tracking data
    akshar_data = load_akshar_tracking()
    duration_data = load_duration_tracking()
    
    current_date = get_current_ist_date()
    
    # Get all unique usernames from both tracking systems
    all_users = set()
    
    # Add users from akshar tracking
    for date_stats in akshar_data["daily"].values():
        all_users.update(date_stats.keys())
    all_users.update(akshar_data["overall"].keys())
    
    # Add users from duration tracking
    for date_stats in duration_data["daily"].values():
        all_users.update(date_stats.keys())
    all_users.update(duration_data["overall"].keys())
    
    # Get list of all registered users from users.json
    users_data = load_users()
    registered_users = [u["username"] for u in users_data]
    
    # Also include registered users even if they have no stats yet
    all_users.update(registered_users)
    
    # Compile stats for each user
    stats = []
    for username in sorted(all_users):
        # Today's stats
        today_akshar = akshar_data["daily"].get(current_date, {}).get(username, 0)
        today_duration = duration_data["daily"].get(current_date, {}).get(username, 0)
        
        # Lifetime stats
        lifetime_akshar = akshar_data["overall"].get(username, 0)
        lifetime_duration = duration_data["overall"].get(username, 0)
        
        # Get completed files count
        completed_files = len(get_user_completed_files(username))
        
        stats.append({
            "username": username,
            "today_akshar": today_akshar,
            "today_duration": round(today_duration, 1),
            "lifetime_akshar": lifetime_akshar,
            "lifetime_duration": round(lifetime_duration, 1),
            "completed_files": completed_files,
            "registered": username in registered_users
        })
    
    # Calculate totals
    totals = {
        "total_users": len(stats),
        "total_today_akshar": sum(s["today_akshar"] for s in stats),
        "total_today_duration": round(sum(s["today_duration"] for s in stats), 1),
        "total_lifetime_akshar": sum(s["lifetime_akshar"] for s in stats),
        "total_lifetime_duration": round(sum(s["lifetime_duration"] for s in stats), 1),
        "total_completed_files": sum(s["completed_files"] for s in stats),
        "date": current_date
    }
    
    return jsonify({
        "stats": stats,
        "totals": totals
    })

@app.route('/api/user-daily-breakdown/<username>')
def get_user_daily_breakdown(username):
    """Get daily breakdown for a specific user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    # Load data
    akshar_data = load_akshar_tracking()
    duration_data = load_duration_tracking()
    
    # Get all dates where user has activity
    all_dates = set()
    
    # From akshar tracking
    for date, users in akshar_data["daily"].items():
        if username in users:
            all_dates.add(date)
    
    # From duration tracking
    for date, users in duration_data["daily"].items():
        if username in users:
            all_dates.add(date)
    
    # Create daily breakdown
    daily_stats = []
    for date in sorted(all_dates, reverse=True):
        daily_stats.append({
            "date": date,
            "akshar": akshar_data["daily"].get(date, {}).get(username, 0),
            "duration": round(duration_data["daily"].get(date, {}).get(username, 0), 1)
        })
    
    return jsonify(daily_stats)


@app.route('/api/remove-user', methods=['POST'])
def remove_user():
    """Remove a user account from users.json"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    username_to_remove = data.get("username")
    
    if not username_to_remove:
        return jsonify({"error": "username required"}), 400
    
    # Load current users
    users = load_users()
    
    # Check if user exists
    user_exists = any(u["username"] == username_to_remove for u in users)
    
    if not user_exists:
        return jsonify({"error": "user not found"}), 404
    
    # Remove user from users list
    users = [u for u in users if u["username"] != username_to_remove]
    
    # Save updated users list
    save_users(users)
    
    return jsonify({
        "message": f"User {username_to_remove} removed successfully",
        "removed_user": username_to_remove
    })



# Initialize file status on startup
# Initialize all tracking on startup
init_file_status()
load_akshar_tracking()
load_duration_tracking()  # Add this line

if __name__ == '__main__':
    print("=" * 50)
    print("WAV Annotation Server - Distributed Mode")
    print("=" * 50)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Annotations folder: {ANNOTATIONS_FOLDER}")
    print(f"Autosave folder: {AUTOSAVE_FOLDER}")
    print(f"Users: {USERS_FILE}")
    print(f"File status: {FILE_STATUS_FILE}")
    print(f"Akshar tracking: {AKSHAR_TRACKING_FILE}")
    print(f"Duration tracking: {DURATION_TRACKING_FILE}")  # Add this line
    print(f"Daily target: {AKSHAR_DAILY_TARGET} akshars")
    print(f"Overall target: {AKSHAR_OVERALL_TARGET} akshars")
    print("=" * 50)
    app.run(debug=False, port=5001, host='0.0.0.0')

