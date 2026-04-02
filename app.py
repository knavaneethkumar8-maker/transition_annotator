from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, session, send_file
from flask_socketio import SocketIO, emit
import os
import json
from datetime import datetime, timedelta, timezone
import glob
import soundfile as sf
import hashlib
import uuid
import pytz
import torch
import numpy as np
import tempfile
import subprocess
import base64
import io
import wave
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = "annotator_secret_key_2024"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=True,
    engineio_logger=True
)

# Folder structure
DATA_FOLDER = 'data'
ANNOTATIONS_FOLDER = "annotations"
AUTOSAVE_FOLDER = "autosave"
USERS_FILE = "users.json"
FILE_STATUS_FILE = "file_status.json"
RECORDINGS_FOLDER = "recordings"  # New folder for VAD recordings
# Track skipped files per user (in memory)
user_skips = {}

COMPANY_PASSWORD = "akshar@123"

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(AUTOSAVE_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)  # Create recordings folder

# ==============================
# VAD MODEL INTEGRATION
# ==============================

# Import VAD modules (adjust paths as needed)
try:
    from vad_model import VADNet
    from vad_utils import extract_feature
    
    device = torch.device("cpu")
    
    model_path = "vad_best.pt"
    if os.path.exists(model_path):
        model = VADNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("VAD Model Loaded Successfully")
    else:
        model = None
        print("VAD model not found")
except ImportError:
    print("VAD modules not found. VAD functionality will be disabled.")
    model = None

def predict_vad(audio_chunk):
    try:
        if model is None:
            return "silence"

        feat = extract_feature(audio_chunk)
        x = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1).item()

        return "speech" if pred == 1 else "silence"

    except Exception as e:
        print(f"Prediction error: {e}")

        # 🔥 IMPORTANT: fallback instead of always silence
        energy = np.mean(audio_chunk ** 2)

        if energy > 1e-6:
            return "speech"
        return "silence"

def convert_audio_to_wav(audio_bytes, original_format='webm'):
    """Convert audio to WAV format with 16kHz sample rate"""
    try:
        with tempfile.NamedTemporaryFile(suffix=f'.{original_format}', delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name
        
        temp_output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # Try using ffmpeg first
        try:
            cmd = [
                'ffmpeg', '-i', temp_input_path,
                '-ar', '16000',
                '-ac', '1',
                '-c', 'pcm_s16le',
                '-y',
                temp_output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            audio_data, samplerate = sf.read(temp_output_path)
            
            # Clean up
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except:
                pass
            
            return audio_data, samplerate
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to soundfile
            try:
                audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
                return audio_data, samplerate
            except:
                raise Exception("Could not convert audio")
                
    except Exception as e:
        print(f"Error in convert_audio_to_wav: {str(e)}")
        raise

# ==============================
# WebSocket Routes for VAD
# ==============================

# Add LIVE_FOLDER definition
LIVE_FOLDER = "live"
os.makedirs(LIVE_FOLDER, exist_ok=True)

# Add client username mapping
client_usernames = {}  # Store mapping of socket_id to username


def get_user_live_folder(username):
    """Get user-specific live folder for streaming chunks"""
    user_folder = os.path.join(LIVE_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder


import threading

def save_chunk_async(audio_chunk, username):
    """Save audio chunk asynchronously to live folder"""
    try:
        # Get user's live folder
        user_folder = get_user_live_folder(username)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"chunk_{timestamp}.wav"
        filepath = os.path.join(user_folder, filename)
        
        # Convert float32 to int16 (16-bit PCM)
        audio_int16 = np.int16(np.clip(audio_chunk, -1.0, 1.0) * 32767)
        
        # Save as WAV file
        sf.write(filepath, audio_int16, 16000, subtype='PCM_16')
        
        print(f"[LIVE] Saved chunk: {filename} for user {username}")
        
    except Exception as e:
        print(f"[LIVE ERROR] Failed to save chunk for {username}: {e}")



@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')
    
    # Clean up mapping
    if request.sid in client_usernames:
        username = client_usernames[request.sid]
        print(f"User {username} disconnected")
        del client_usernames[request.sid]

@socketio.on('register_user')
def handle_register_user(data):
    """Register a user with their socket connection"""
    username = data.get('username')
    if username:
        client_usernames[request.sid] = username
        print(f"Registered client {request.sid} as user: {username}")
        emit('user_registered', {'status': 'success', 'username': username})
    else:
        print(f"No username provided for client {request.sid}")

@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        # Get username from mapping
        username = client_usernames.get(request.sid)
        
        # Decode base64
        audio_bytes = base64.b64decode(data['audio'])
        
        # Convert to float32 numpy
        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if len(audio_chunk) == 0:
            return
        
        # Safety: ensure correct chunk size (54ms = 864 samples @16kHz)
        if len(audio_chunk) != 864:
            print(f"Invalid chunk size: {len(audio_chunk)}")
            return
        
        # Predict VAD
        prediction = predict_vad(audio_chunk)
        
        # Send prediction back to client
        emit('vad_prediction', {
            'result': prediction,
            'timestamp': data.get('timestamp', 0)
        })
        
        # Save chunk to live folder if username is available
        if username:
            # Save asynchronously to avoid blocking
            threading.Thread(
                target=save_chunk_async,
                args=(audio_chunk.copy(), username),
                daemon=True
            ).start()
        else:
            print(f"No username for client {request.sid}, skipping save")
            
    except Exception as e:
        print(f"Error processing audio stream: {e}")
        import traceback
        traceback.print_exc()




@app.route('/api/live-status')
def live_status():
    """Check live folder status for current user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    user_folder = get_user_live_folder(username)
    
    # Count files in user's live folder
    files = []
    if os.path.exists(user_folder):
        files = [f for f in os.listdir(user_folder) if f.endswith('.wav')]
    
    return jsonify({
        "username": username,
        "folder": user_folder,
        "exists": os.path.exists(user_folder),
        "file_count": len(files),
        "files": files[-10:]  # Show last 10 files
    })


# ==============================
# NEW: Category Management
# ==============================

def get_available_categories():
    """Get all top-level subfolders in data folder as categories"""
    categories = []
    try:
        for item in os.listdir(DATA_FOLDER):
            item_path = os.path.join(DATA_FOLDER, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                categories.append(item)
    except Exception as e:
        print(f"Error reading categories: {e}")
    return sorted(categories)

def get_files_by_category(category=None):
    """
    Get all _4x.wav files from a specific category or all categories
    Returns list of tuples (filename, category)
    """
    files = []
    
    if category is None or category == 'all':
        # Get all files from all subfolders
        for root, dirs, files_in_dir in os.walk(DATA_FOLDER):
            for file in files_in_dir:
                if file.endswith('_4x.wav'):
                    # Get relative path to determine category
                    rel_path = os.path.relpath(root, DATA_FOLDER)
                    if rel_path == '.':
                        # Files directly in data folder
                        file_category = 'root'
                    else:
                        file_category = rel_path.split(os.sep)[0]  # Top-level folder
                    files.append((file, file_category))
    else:
        # Get files only from specific category folder
        category_path = os.path.join(DATA_FOLDER, category)
        if os.path.exists(category_path):
            for root, dirs, files_in_dir in os.walk(category_path):
                for file in files_in_dir:
                    if file.endswith('_4x.wav'):
                        files.append((file, category))
    
    return files

def get_user_category_preference(username):
    """Get user's current category preference from session"""
    # Store in session instead of file for simplicity
    return session.get(f'category_{username}', 'all')

def set_user_category_preference(username, category):
    """Set user's category preference"""
    session[f'category_{username}'] = category

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

        if tracking.get("last_reset") != current_date:
            tracking["daily"] = {}
            tracking["last_reset"] = current_date
            save_akshar_tracking(tracking)

        return tracking
    else:
        return init_akshar_tracking()
    
def init_akshar_tracking():
    """Initialize akshar tracking structure"""
    tracking = {
        "daily": {},
        "overall": {},
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
    akshar_count = sum(1 for frame in frames if frame.get("text") and frame["text"].strip() != "")
    
    if akshar_count == 0:
        return {"added": 0}
    
    tracking = load_akshar_tracking()
    current_date = get_current_ist_date()
    
    if current_date not in tracking["daily"]:
        tracking["daily"][current_date] = {}
    
    tracking["daily"][current_date][username] = tracking["daily"][current_date].get(username, 0) + akshar_count
    tracking["overall"][username] = tracking["overall"].get(username, 0) + akshar_count
    
    save_akshar_tracking(tracking)
    
    return {
        "added": akshar_count,
        "user_daily": tracking["daily"][current_date][username],
        "user_overall": tracking["overall"][username]
    }

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

        if tracking.get("last_reset") != current_date:
            tracking["daily"] = {}
            tracking["last_reset"] = current_date
            save_duration_tracking(tracking)

        return tracking
    else:
        return init_duration_tracking()

def init_duration_tracking():
    """Initialize duration tracking structure"""
    tracking = {
        "daily": {},
        "overall": {},
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
    
    if current_date not in tracking["daily"]:
        tracking["daily"][current_date] = {}
    
    tracking["daily"][current_date][username] = tracking["daily"][current_date].get(username, 0) + duration_seconds
    tracking["overall"][username] = tracking["overall"].get(username, 0) + duration_seconds
    
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
    
    daily_stats = tracking["daily"].get(current_date, {})
    total_daily = sum(daily_stats.values())
    
    user_daily = daily_stats.get(username, 0)
    user_overall = tracking["overall"].get(username, 0)
    total_overall = sum(tracking["overall"].values())
    
    return {
        "date": current_date,
        "user_daily": round(user_daily, 1),
        "user_overall": round(user_overall, 1),
        "total_daily": round(total_daily, 1),
        "total_overall": round(total_overall, 1)
    }

def get_akshar_stats(username):
    """Get simple akshar statistics"""
    tracking = load_akshar_tracking()
    current_date = get_current_ist_date()
    
    daily_stats = tracking["daily"].get(current_date, {})
    total_daily = sum(daily_stats.values())
    
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
    user_autosave_dir = os.path.join(AUTOSAVE_FOLDER, username)
    os.makedirs(user_autosave_dir, exist_ok=True)
    
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
    # Get all files with their categories
    all_files_with_categories = get_files_by_category()
    
    # Load existing file status if it exists
    existing_status = {}
    if os.path.exists(FILE_STATUS_FILE):
        try:
            with open(FILE_STATUS_FILE, 'r', encoding='utf-8') as f:
                existing_status = json.load(f)
        except Exception as e:
            print("Error reading existing file status:", e)
            existing_status = {}

    new_files = []
    updated_status = existing_status.copy()

    for filename, category in all_files_with_categories:
        if filename not in updated_status:
            print(f"Adding new file: {filename} (category: {category})")
            new_files.append(filename)
            updated_status[filename] = {
                "status": "pending",
                "assigned_to": None,
                "assigned_at": None,
                "completed_at": None,
                "annotation_file": None,
                "priority": 1 if filename.startswith('BEEJ_') else 0,
                "category": category  # Store category in file status
            }
        else:
            # Ensure category exists for existing files
            if "category" not in updated_status[filename]:
                updated_status[filename]["category"] = category
            # Ensure priority flag exists
            if "priority" not in updated_status[filename]:
                updated_status[filename]["priority"] = 1 if filename.startswith('BEEJ_') else 0

    if new_files:
        print(f"Added {len(new_files)} new files to tracking")
        save_file_status(updated_status)
    elif updated_status != existing_status:
        save_file_status(updated_status)

    return updated_status

def find_audio_file(filename):
    """Find full path of a file inside DATA_FOLDER recursively"""
    matches = glob.glob(os.path.join(DATA_FOLDER, '**', filename), recursive=True)
    return matches[0] if matches else None

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
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "audio_file" in data:
                        completed_files.append(data["audio_file"])
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    return completed_files

def get_next_file_for_user(username, category=None):
    """
    Get next unassigned or pending file for user, with optional category filter
    If category is None or 'all', consider all files
    """
    file_status = init_file_status()
    user_completed = get_user_completed_files(username)
    skipped_files = user_skips.get(username, [])
    
    # If user already has assigned file
    for filename, status in file_status.items():
        if status.get("assigned_to") == username and status.get("status") == "assigned":
            # Check if file matches category filter
            if category is None or category == 'all' or status.get("category") == category:
                if filename not in user_completed:
                    return filename
    
    # Filter by category if specified
    filtered_files = []
    for filename, status in file_status.items():
        if category is None or category == 'all' or status.get("category") == category:
            filtered_files.append((filename, status))
    
    # Check for available BEEJ_ files
    available_beej_files = []
    for filename, status in filtered_files:
        if (status.get("priority", 0) == 1 and 
            status.get("status") == "pending" and 
            filename not in user_completed and
            filename not in skipped_files):
            available_beej_files.append((filename, status))
    
    if available_beej_files:
        available_beej_files.sort(key=lambda x: x[0])
        filename, status = available_beej_files[0]
        status["status"] = "assigned"
        status["assigned_to"] = username
        status["assigned_at"] = datetime.now().isoformat()
        save_file_status(file_status)
        return filename
    
    # Get pending regular files
    pending_regular_files = []
    for filename, status in filtered_files:
        if (status.get("priority", 0) != 1 and 
            status.get("status") == "pending" and 
            filename not in user_completed and 
            filename not in skipped_files):
            pending_regular_files.append((filename, status))
    
    if pending_regular_files:
        filename, status = pending_regular_files[0]
        status["status"] = "assigned"
        status["assigned_to"] = username
        status["assigned_at"] = datetime.now().isoformat()
        save_file_status(file_status)
        return filename
    
    # Reset skips and try again
    if username in user_skips:
        user_skips[username] = []
    
    # Final attempt: any pending file in filtered list
    for filename, status in filtered_files:
        if status.get("status") == "pending" and filename not in user_completed:
            status["status"] = "assigned"
            status["assigned_to"] = username
            status["assigned_at"] = datetime.now().isoformat()
            save_file_status(file_status)
            return filename
    
    return None

def get_category_progress(username, category):
    """Get progress statistics for a specific category"""
    file_status = init_file_status()
    user_completed = get_user_completed_files(username)
    
    total = 0
    completed = 0
    pending = 0
    
    for filename, status in file_status.items():
        if status.get("category") == category:
            total += 1
            if filename in user_completed or status.get("status") == "completed":
                completed += 1
            elif status.get("status") == "pending":
                pending += 1
    
    return {
        "total": total,
        "completed": completed,
        "pending": pending
    }

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
    phone = data.get("phone")
    company_password = data.get("company_password")

    # 🔒 Company password check
    if company_password != COMPANY_PASSWORD:
        return jsonify({"error": "Invalid company password"}), 403

    # 📱 Basic phone validation (optional but recommended)
    if not phone or len(phone) < 8:
        return jsonify({"error": "Invalid phone number"}), 400

    users = load_users()

    if any(u["username"] == username for u in users):
        return jsonify({"error": "User exists"}), 400

    users.append({
        "username": username,
        "password": hash_password(password),
        "phone": phone
    })

    save_users(users)
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
            # Initialize category preference
            if f'category_{username}' not in session:
                session[f'category_{username}'] = 'all'
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
# NEW: Category Routes
# ==============================

@app.route('/api/categories')
def get_categories():
    """Get available data categories"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    categories = get_available_categories()
    return jsonify({
        "categories": categories,
        "current": get_user_category_preference(session["user"])
    })

@app.route('/api/set-category', methods=['POST'])
def set_category():
    """Set user's current category preference"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    category = data.get("category", "all")
    username = session["user"]
    
    # Validate category
    if category != "all" and category not in get_available_categories():
        return jsonify({"error": "Invalid category"}), 400
    
    set_user_category_preference(username, category)
    
    # Clear user's current assignment when switching categories
    file_status = init_file_status()
    for filename, status in file_status.items():
        if status.get("assigned_to") == username and status.get("status") == "assigned":
            status["status"] = "pending"
            status["assigned_to"] = None
            status["assigned_at"] = None
    
    save_file_status(file_status)
    
    return jsonify({
        "message": f"Switched to {category}",
        "category": category
    })

@app.route('/api/category-progress/<category>')
def category_progress(category):
    """Get progress for a specific category"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    stats = get_category_progress(username, category)
    return jsonify(stats)

# ==============================
# Find matching normal WAV file
# ==============================
@app.route('/api/matching-wav/<filename>')
def get_matching_wav(filename):
    """Find a matching normal WAV file for comparison"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    normal_filename = filename.replace('_4x', '')
    normal_path = find_audio_file(normal_filename)
    
    if normal_path and os.path.exists(normal_path):
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
    filepath = find_audio_file(filename)
    if not filepath:
        return redirect("/login")

    directory = os.path.dirname(filepath)
    file_only = os.path.basename(filepath)

    return send_from_directory(directory, file_only)

# ==============================
# ROUTES
# ==============================

@app.route("/")
def index():
    if not require_login():
        return redirect("/login")
    
    # Get available categories for the template
    categories = get_available_categories()
    
    return render_template("index.html", user=session["user"], categories=categories)

@app.route('/api/next-file')
def get_next_file():
    """Get the next file to annotate for the current user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    category = get_user_category_preference(username)
    next_file = get_next_file_for_user(username, category)
    
    if next_file:
        return jsonify({
            "filename": next_file,
            "message": "File assigned successfully"
        })
    else:
        return jsonify({
            "filename": None,
            "message": f"No more files to annotate in {category}"
        })

@app.route('/api/current-file')
def get_current_file():
    """Get the currently assigned file for user (if any)"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    category = get_user_category_preference(username)
    file_status = init_file_status()
    
    for filename, status in file_status.items():
        if (status["assigned_to"] == username and 
            status["status"] == "assigned" and
            (category == 'all' or status.get("category") == category)):
            return jsonify({"filename": filename})
    
    return jsonify({"filename": None})

@app.route('/api/file-info/<filename>')
def get_file_info(filename):
    """Get information about a specific file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    filepath = find_audio_file(filename)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "file not found"}), 404
    
    info = sf.info(filepath)
    
    duration = info.duration
    sr = info.samplerate
    samples = info.frames
    
    base = os.path.splitext(filename)[0]
    # Find JSON file in same directory as audio
    json_file = os.path.join(os.path.dirname(filepath), base + ".json")
    
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
    filepath = find_audio_file(filename)
    if not filepath:
        return redirect("/login")

    directory = os.path.dirname(filepath)
    file_only = os.path.basename(filepath)

    return send_from_directory(directory, file_only)

@app.route('/api/phn/<filename>')
def get_phn(filename):
    """Get PHN annotations for a file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    base = os.path.splitext(filename)[0]
    filepath = find_audio_file(filename)
    
    if not filepath:
        return jsonify([])
    
    phn_file = os.path.join(os.path.dirname(filepath), base + ".PHN")
    
    if not os.path.exists(phn_file):
        return jsonify([])
    
    info = sf.info(filepath)
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
    filepath = find_audio_file(filename)
    
    if not filepath:
        return jsonify({"frames": [], "sentence": ""})
    
    json_file = os.path.join(os.path.dirname(filepath), base + ".json")
    
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
    category = get_user_category_preference(username)
    file_status = init_file_status()
    
    # Filter by category if needed
    filtered_files = {}
    for filename, status in file_status.items():
        if category == 'all' or status.get("category") == category:
            filtered_files[filename] = status
    
    total_files = len(filtered_files)
    
    # Global completed within category
    global_completed_count = sum(
        1 for f in filtered_files.values()
        if f["status"] == "completed"
    )
    
    # User completed within category
    user_completed_count = sum(
        1 for f in filtered_files.values()
        if f["status"] == "completed"
        and f["assigned_to"] == username
    )
    
    # Current file within category
    current_file = None
    for filename, status in filtered_files.items():
        if status["assigned_to"] == username and status["status"] == "assigned":
            current_file = filename
            break
    
    return jsonify({
        "username": username,
        "category": category,
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
    category = get_user_category_preference(username)
    
    # Get file duration
    filepath = find_audio_file(audio_file)
    duration_seconds = 0
    if filepath and os.path.exists(filepath):
        info = sf.info(filepath)
        duration_seconds = info.duration
    
    # Update akshar counts
    akshar_update = update_akshar_counts(username, frames)
    
    # Update duration counts
    duration_update = update_duration_counts(username, duration_seconds)
    
    # Filename (NO timestamp)
    base = os.path.splitext(audio_file)[0]
    annotation_filename = f"{base}.json"
    
    # Save JSON
    user_dir = get_user_annotation_dir(username)
    output_file = os.path.join(user_dir, annotation_filename)
    
    output_data = {
        "audio_file": audio_file,
        "annotator": username,
        "timestamp": datetime.now().isoformat(),
        "window_ms": data.get("window_ms"),
        "sentence": data.get("sentence"),
        "full_sequence": data.get("full_sequence"),
        "frames": frames,
        "category": category
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # =========================
    # 🔥 TEXTGRID GENERATION
    # =========================
    
    def create_textgrid(frames, duration, sentence, annotator):
        tg = []

        tg.append('File type = "ooTextFile"')
        tg.append('Object class = "TextGrid"\n')

        tg.append(f"xmin = 0")
        tg.append(f"xmax = {duration}")
        tg.append("tiers? <exists>")
        tg.append("size = 3")
        tg.append("item []:")

        # sentence tier
        tg.append("    item [1]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "sentence"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append("        intervals: size = 1")

        tg.append("        intervals [1]:")
        tg.append(f"            xmin = 0")
        tg.append(f"            xmax = {duration}")
        tg.append(f'            text = "{sentence}"')

        # annotations tier
        tg.append("    item [2]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "annotations"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append(f"        intervals: size = {len(frames)}")

        for i, f in enumerate(frames, 1):
            start = f["start_ms"] / 1000.0
            end = f["end_ms"] / 1000.0
            text = f["text"] if f["text"] else ""

            tg.append(f"        intervals [{i}]:")
            tg.append(f"            xmin = {start}")
            tg.append(f"            xmax = {end}")
            tg.append(f'            text = "{text}"')

        # annotator tier
        tg.append("    item [3]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "annotator"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append("        intervals: size = 1")

        tg.append("        intervals [1]:")
        tg.append(f"            xmin = 0")
        tg.append(f"            xmax = {duration}")
        tg.append(f'            text = "{annotator}"')

        return "\n".join(tg)

    def scale_frames(frames, factor):
        return [
            {
                "start_ms": f["start_ms"] / factor,
                "end_ms": f["end_ms"] / factor,
                "text": f["text"]
            }
            for f in frames
        ]

    # 🔹 4x TG
    duration_4x = frames[-1]["end_ms"] / 1000.0 if frames else 0
    tg_4x = create_textgrid(frames, duration_4x, data.get("full_sequence", ""), username)

    tg_4x_path = os.path.join(user_dir, f"{base}.TextGrid")
    with open(tg_4x_path, "w", encoding="utf-8") as f:
        f.write(tg_4x)

    # 🔹 NORMAL TG
    normal_frames = scale_frames(frames, 4)
    normal_duration = normal_frames[-1]["end_ms"] / 1000.0 if normal_frames else 0

    normal_base = base.replace("_4x", "")
    tg_normal = create_textgrid(normal_frames, normal_duration, data.get("full_sequence", ""), username)

    tg_normal_path = os.path.join(user_dir, f"{normal_base}.TextGrid")
    with open(tg_normal_path, "w", encoding="utf-8") as f:
        f.write(tg_normal)

    # =========================
    # 🔥 SAVE TO UI_DATASET
    # =========================
    UI_DATASET_DIR = "UI_DATASET"
    os.makedirs(UI_DATASET_DIR, exist_ok=True)

    ui_tg_path = os.path.join(UI_DATASET_DIR, f"{normal_base}.TextGrid")
    with open(ui_tg_path, "w", encoding="utf-8") as f:
        f.write(tg_normal)

    # =========================
    # FINAL STEPS
    # =========================

    mark_file_completed(audio_file, username, annotation_filename)

    autosave_path = get_autosave_path(username, audio_file)
    if os.path.exists(autosave_path):
        os.remove(autosave_path)

    next_file = get_next_file_for_user(username, category)

    return jsonify({
        "message": "saved",
        "file": annotation_filename,
        "next_file": next_file,
        "akshar": akshar_update,
        "duration": duration_update
    })

@app.route('/api/skip-file', methods=['POST'])
def skip_file():
    """Skip current file and get next one"""
    if not require_login():
        return jsonify({"error": "login required"}), 401

    username = session["user"]
    data = request.json
    current_file = data.get("current_file")
    category = get_user_category_preference(username)

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

    is_beej = current_file.startswith('BEEJ_') if current_file else False
    
    next_file = get_next_file_for_user(username, category)
    
    message = "file skipped"
    if is_beej and not next_file:
        file_status = init_file_status()
        beej_in_progress = False
        for filename, status in file_status.items():
            if (status.get("priority", 0) == 1 and 
                status.get("status") == "assigned" and 
                status.get("assigned_to") != username and
                (category == 'all' or status.get("category") == category)):
                beej_in_progress = True
                break
        
        if beej_in_progress:
            message = "BEEJ_ files are being processed by others. Please wait."

    return jsonify({
        "message": message,
        "next_file": next_file
    })

# ==============================
# ADMIN STATS PAGE (Unchanged)
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
    
    akshar_data = load_akshar_tracking()
    duration_data = load_duration_tracking()
    
    current_date = get_current_ist_date()
    
    all_users = set()
    
    for date_stats in akshar_data["daily"].values():
        all_users.update(date_stats.keys())
    all_users.update(akshar_data["overall"].keys())
    
    for date_stats in duration_data["daily"].values():
        all_users.update(date_stats.keys())
    all_users.update(duration_data["overall"].keys())
    
    users_data = load_users()
    registered_users = [u["username"] for u in users_data]
    all_users.update(registered_users)
    
    stats = []
    for username in sorted(all_users):
        today_akshar = akshar_data["daily"].get(current_date, {}).get(username, 0)
        today_duration = duration_data["daily"].get(current_date, {}).get(username, 0)
        
        lifetime_akshar = akshar_data["overall"].get(username, 0)
        lifetime_duration = duration_data["overall"].get(username, 0)
        
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
    
    akshar_data = load_akshar_tracking()
    duration_data = load_duration_tracking()
    
    all_dates = set()
    
    for date, users in akshar_data["daily"].items():
        if username in users:
            all_dates.add(date)
    
    for date, users in duration_data["daily"].items():
        if username in users:
            all_dates.add(date)
    
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
    
    users = load_users()
    user_exists = any(u["username"] == username_to_remove for u in users)
    
    if not user_exists:
        return jsonify({"error": "user not found"}), 404
    
    users = [u for u in users if u["username"] != username_to_remove]
    save_users(users)
    
    return jsonify({
        "message": f"User {username_to_remove} removed successfully",
        "removed_user": username_to_remove
    })

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


@app.route('/career/jobs')
def jobs_page():
    return render_template("jobs.html")

@app.route('/career/jobs/annotator')
def job_portal():
    return render_template("job_annotator.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404



# ==============================
# VAD Recording Routes
# ==============================

# ==============================
# VAD Recording Routes with User Folders
# ==============================
# ==============================
# VAD Recording Routes with User Folders
# ==============================

def get_user_recordings_folder(username):
    """Get user-specific recordings folder, create if not exists"""
    # User folder is inside the main recordings folder
    user_folder = os.path.join(RECORDINGS_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

@app.route("/vad-recorder")
def vad_recorder():
    """VAD Recording page"""
    if not require_login():
        return redirect("/login")
    
    # Get username from session and pass to template
    username = session.get("user")
    return render_template("vad_recorder.html", user=username)

@app.route("/api/vad/save_recording", methods=["POST"])
def save_vad_recording():
    """Save VAD recorded audio in user-specific folder"""
    try:
        if not require_login():
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        file = request.files['audio']
        username = session["user"]
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Get the file extension
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'webm'
        
        # Read the audio data
        audio_bytes = file.read()
        
        if len(audio_bytes) == 0:
            return jsonify({"success": False, "error": "Empty audio file"}), 400
        
        # Convert audio to WAV format
        audio_data, samplerate = convert_audio_to_wav(audio_bytes, file_extension)
        
        # Ensure audio data is in the correct range
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Generate filename with username as prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"{username}_{timestamp}.wav"
        
        # Save ONLY in user-specific folder (not in main recordings folder)
        user_folder = get_user_recordings_folder(username)
        filepath = os.path.join(user_folder, filename)
        
        # Save the file
        sf.write(filepath, audio_data, 16000, subtype='PCM_16')
        
        # Get file duration
        duration = len(audio_data) / 16000
        
        # Update duration tracking for VAD recordings
        update_duration_counts(username, duration)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "duration": round(duration, 2),
            "message": "Recording saved successfully"
        })
            
    except Exception as e:
        print(f"Error saving recording: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/vad/get_recordings", methods=["GET"])
def get_vad_recordings():
    """Get list of VAD recordings for the current user from their folder"""
    try:
        if not require_login():
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        username = session["user"]
        user_folder = get_user_recordings_folder(username)
        files = []
        
        if os.path.exists(user_folder):
            for filename in os.listdir(user_folder):
                # Match files that start with username_ and end with .wav
                if filename.endswith('.wav') and filename.startswith(f"{username}_"):
                    filepath = os.path.join(user_folder, filename)
                    try:
                        stat = os.stat(filepath)
                        info = sf.info(filepath)
                        # Extract timestamp from filename for display
                        # Remove username_ prefix and .wav suffix
                        display_name = filename.replace(f"{username}_", "").replace(".wav", "")
                        files.append({
                            "filename": filename,
                            "name": display_name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "duration": round(info.duration, 2)
                        })
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
                        continue
            
            # Sort by modified time (newest first)
            files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({"success": True, "files": files})
    except Exception as e:
        print(f"Error getting recordings: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/vad/play_recording/<filename>", methods=["GET"])
def play_vad_recording(filename):
    """Play a VAD recording from user's folder"""
    try:
        if not require_login():
            return redirect("/login")
        
        username = session["user"]
        user_folder = get_user_recordings_folder(username)
        
        # Security: prevent directory traversal
        filename = secure_filename(filename)
        filepath = os.path.join(user_folder, filename)
        
        if os.path.exists(filepath):
            try:
                return send_file(filepath, mimetype='audio/wav', conditional=True, as_attachment=False)
            except Exception as e:
                print(f"Error sending file: {e}")
                return jsonify({"success": False, "error": "Error serving file"}), 500
        else:
            return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        print(f"Error playing recording: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/vad/delete_recording/<filename>", methods=["DELETE"])
def delete_vad_recording(filename):
    """Delete a VAD recording from user's folder"""
    try:
        if not require_login():
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        username = session["user"]
        user_folder = get_user_recordings_folder(username)
        
        # Security: prevent directory traversal
        filename = secure_filename(filename)
        filepath = os.path.join(user_folder, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True, "message": "Recording deleted successfully"})
        else:
            return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        print(f"Error deleting recording: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# Add this import at the top if not already present
import zipfile
from io import BytesIO

# Add this route after your existing routes
@app.route('/downloads')
def downloads_page():
    """Page for downloading completed annotations"""
    if not require_login():
        return redirect("/login")
    
    # Get all annotators who have completed files
    users = []
    if os.path.exists(ANNOTATIONS_FOLDER):
        users = [d for d in os.listdir(ANNOTATIONS_FOLDER) 
                if os.path.isdir(os.path.join(ANNOTATIONS_FOLDER, d))]
    
    return render_template("downloads.html", user=session["user"], users=sorted(users))

@app.route('/api/user-files/<username>')
def get_user_files(username):
    """Get list of completed files for a specific user"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    user_dir = os.path.join(ANNOTATIONS_FOLDER, username)
    if not os.path.exists(user_dir):
        return jsonify({"files": []})
    
    files_info = []
    
    for json_file in glob.glob(os.path.join(user_dir, "*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            audio_filename = data.get("audio_file", "")
            base_name = os.path.splitext(audio_filename)[0]
            
            # Remove _4x suffix for normal files
            normal_base = base_name.replace("_4x", "")
            
            # Check if files exist
            audio_path = find_audio_file(audio_filename)
            normal_audio_path = find_audio_file(normal_base + ".wav") if normal_base != base_name else None
            
            # TextGrid files in user's annotation folder
            tg_4x_path = os.path.join(user_dir, f"{base_name}.TextGrid")
            tg_normal_path = os.path.join(user_dir, f"{normal_base}.TextGrid") if normal_base != base_name else None
            
            # Check if UI_DATASET has the file (optional)
            ui_tg_path = os.path.join("UI_DATASET", f"{normal_base}.TextGrid")
            
            file_info = {
                "audio_file": audio_filename,
                "base_name": base_name,
                "normal_base": normal_base if normal_base != base_name else None,
                "has_audio": audio_path is not None,
                "has_normal_audio": normal_audio_path is not None,
                "has_tg_4x": os.path.exists(tg_4x_path),
                "has_tg_normal": os.path.exists(tg_normal_path) if tg_normal_path else False,
                "has_ui_tg": os.path.exists(ui_tg_path),
                "timestamp": data.get("timestamp", ""),
                "sentence": data.get("sentence", ""),
                "category": data.get("category", "")
            }
            
            files_info.append(file_info)
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    # Sort by timestamp (newest first)
    files_info.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return jsonify({"files": files_info, "username": username})

@app.route('/api/download-file/<username>/<file_type>/<filename>')
def download_file(username, file_type, filename):
    """Download specific file for a user"""
    if not require_login():
        return redirect("/login")
    
    file_path = None
    
    if file_type == "audio":
        # 4x audio from data folder
        file_path = find_audio_file(filename)
    elif file_type == "normal_audio":
        # Normal audio from data folder
        file_path = find_audio_file(filename)
    elif file_type == "tg_4x":
        # 4x TextGrid from user's annotation folder
        file_path = os.path.join(ANNOTATIONS_FOLDER, username, filename)
    elif file_type == "tg_normal":
        # Normal TextGrid from user's annotation folder
        file_path = os.path.join(ANNOTATIONS_FOLDER, username, filename)
    elif file_type == "ui_tg":
        # UI_DATASET TextGrid
        file_path = os.path.join("UI_DATASET", filename)
    
    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    
    return jsonify({"error": "File not found"}), 404

@app.route('/api/download-batch/<username>', methods=['POST'])
def download_batch(username):
    """Download multiple files as a zip archive"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    files_to_download = data.get("files", [])
    
    if not files_to_download:
        return jsonify({"error": "No files selected"}), 400
    
    # Create zip file in memory
    memory_file = BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_info in files_to_download:
            file_type = file_info.get("type")
            filename = file_info.get("filename")
            display_name = file_info.get("display_name", filename)
            
            file_path = None
            
            if file_type == "audio":
                file_path = find_audio_file(filename)
            elif file_type == "normal_audio":
                file_path = find_audio_file(filename)
            elif file_type == "tg_4x":
                file_path = os.path.join(ANNOTATIONS_FOLDER, username, filename)
            elif file_type == "tg_normal":
                file_path = os.path.join(ANNOTATIONS_FOLDER, username, filename)
            elif file_type == "ui_tg":
                file_path = os.path.join("UI_DATASET", filename)
            
            if file_path and os.path.exists(file_path):
                zf.write(file_path, display_name)
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        download_name=f"{username}_annotations.zip",
        as_attachment=True,
        mimetype='application/zip'
    )




# ==============================
# VERIFICATION PAGE ROUTES
# ==============================

@app.route('/verify')
def verify_page():
    """Verification page for completed annotations"""
    if not require_login():
        return redirect("/login")
    
    # Load all annotators from users.json
    users = load_users()
    annotators = [u["username"] for u in users]
    
    return render_template("verify.html", user=session["user"], annotators=sorted(annotators))

@app.route('/api/annotator-files/<username>')
def get_annotator_files(username):
    """Get all completed files for a specific annotator"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    # Try exact match first, then stripped match
    user_dir = os.path.join(ANNOTATIONS_FOLDER, username)
    if not os.path.exists(user_dir):
        try:
            all_dirs = os.listdir(ANNOTATIONS_FOLDER)
            match = next((d for d in all_dirs if d.strip() == username.strip()), None)
            if match:
                user_dir = os.path.join(ANNOTATIONS_FOLDER, match)
            else:
                return jsonify({"files": []})
        except Exception:
            return jsonify({"files": []})
    
    file_status = init_file_status()
    files_info = []
    
    for json_file in glob.glob(os.path.join(user_dir, "*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            audio_filename = data.get("audio_file", "")
            
            is_verified = False
            if audio_filename in file_status:
                is_verified = file_status[audio_filename].get("verified", False)
            
            files_info.append({
                "filename": audio_filename,
                "annotation_file": os.path.basename(json_file),
                "timestamp": data.get("timestamp", ""),
                "sentence": data.get("sentence", ""),
                "verified": is_verified,
                "annotator": username
            })
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    files_info.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return jsonify({"files": files_info, "username": username})

@app.route('/api/load-for-verification/<username>/<filename>')
def load_for_verification(username, filename):
    """Load a completed annotation for verification"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    # Load the completed annotation JSON from annotator's folder
    annotation_path = os.path.join(ANNOTATIONS_FOLDER, username, f"{os.path.splitext(filename)[0]}.json")
    
    if not os.path.exists(annotation_path):
        return jsonify({"error": "Annotation file not found"}), 404
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # Get audio file path
        audio_path = find_audio_file(filename)
        
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404
        
        # Get audio info
        info = sf.info(audio_path)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "frames": annotation_data.get("frames", []),
            "sentence": annotation_data.get("sentence", ""),
            "full_sequence": annotation_data.get("full_sequence", ""),
            "annotator": username,
            "timestamp": annotation_data.get("timestamp", "")
        })
        
    except Exception as e:
        print(f"Error loading for verification: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/verify-submit', methods=['POST'])
def verify_submit():
    """Submit verification for a file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    filename = data.get("filename")
    username = data.get("annotator")
    frames = data.get("frames", [])
    verified_by = session["user"]
    
    if not filename or not username:
        return jsonify({"error": "Missing data"}), 400
    
    # Create verified folder
    VERIFIED_FOLDER = "verified"
    os.makedirs(VERIFIED_FOLDER, exist_ok=True)
    
    # Save verified annotation - NO _verified suffix
    base = os.path.splitext(filename)[0]
    verified_filename = f"{base}.json"
    verified_path = os.path.join(VERIFIED_FOLDER, verified_filename)
    
    verified_data = {
        "audio_file": filename,
        "original_annotator": username,
        "verified_by": verified_by,
        "verified_at": datetime.now().isoformat(),
        "frames": frames,
        "sentence": data.get("sentence", ""),
        "full_sequence": data.get("full_sequence", "")
    }
    
    with open(verified_path, 'w', encoding='utf-8') as f:
        json.dump(verified_data, f, indent=2, ensure_ascii=False)
    
    # =========================
    # 🔥 TEXTGRID GENERATION FOR VERIFIED FILES
    # =========================
    
    def create_textgrid(frames, duration, sentence, annotator):
        tg = []

        tg.append('File type = "ooTextFile"')
        tg.append('Object class = "TextGrid"\n')

        tg.append(f"xmin = 0")
        tg.append(f"xmax = {duration}")
        tg.append("tiers? <exists>")
        tg.append("size = 3")
        tg.append("item []:")

        # sentence tier
        tg.append("    item [1]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "sentence"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append("        intervals: size = 1")

        tg.append("        intervals [1]:")
        tg.append(f"            xmin = 0")
        tg.append(f"            xmax = {duration}")
        tg.append(f'            text = "{sentence}"')

        # annotations tier
        tg.append("    item [2]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "annotations"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append(f"        intervals: size = {len(frames)}")

        for i, f in enumerate(frames, 1):
            start = f["start_ms"] / 1000.0
            end = f["end_ms"] / 1000.0
            text = f["text"] if f["text"] else ""

            tg.append(f"        intervals [{i}]:")
            tg.append(f"            xmin = {start}")
            tg.append(f"            xmax = {end}")
            tg.append(f'            text = "{text}"')

        # annotator tier
        tg.append("    item [3]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "annotator"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append("        intervals: size = 1")

        tg.append("        intervals [1]:")
        tg.append(f"            xmin = 0")
        tg.append(f"            xmax = {duration}")
        tg.append(f'            text = "{annotator}"')

        return "\n".join(tg)

    def scale_frames(frames, factor):
        return [
            {
                "start_ms": f["start_ms"] / factor,
                "end_ms": f["end_ms"] / factor,
                "text": f["text"]
            }
            for f in frames
        ]

    # Get audio file duration for 4x version
    audio_path = find_audio_file(filename)
    if audio_path and os.path.exists(audio_path):
        info = sf.info(audio_path)
        duration_4x = info.duration
    else:
        duration_4x = frames[-1]["end_ms"] / 1000.0 if frames else 0

    # 🔹 Generate 4x TextGrid (slow version) - NO _verified suffix
    tg_4x = create_textgrid(frames, duration_4x, data.get("full_sequence", ""), username)
    
    # Save 4x TextGrid in verified folder
    tg_4x_filename = f"{base}.TextGrid"
    tg_4x_path = os.path.join(VERIFIED_FOLDER, tg_4x_filename)
    with open(tg_4x_path, "w", encoding="utf-8") as f:
        f.write(tg_4x)

    # 🔹 Generate Normal TextGrid (scale frames by factor 4) - NO _verified suffix
    normal_frames = scale_frames(frames, 4)
    normal_base = base.replace("_4x", "")
    
    # Get normal audio file duration
    normal_audio_path = find_audio_file(normal_base + ".wav")
    if normal_audio_path and os.path.exists(normal_audio_path):
        info = sf.info(normal_audio_path)
        normal_duration = info.duration
    else:
        normal_duration = normal_frames[-1]["end_ms"] / 1000.0 if normal_frames else 0
    
    tg_normal = create_textgrid(normal_frames, normal_duration, data.get("full_sequence", ""), username)
    
    # Save normal TextGrid in verified folder
    tg_normal_filename = f"{normal_base}.TextGrid"
    tg_normal_path = os.path.join(VERIFIED_FOLDER, tg_normal_filename)
    with open(tg_normal_path, "w", encoding="utf-8") as f:
        f.write(tg_normal)

    # =========================
    # 🔥 SAVE TO UI_DATASET (normal version) - NO _verified suffix
    # =========================
    UI_DATASET_DIR = "UI_DATASET"
    os.makedirs(UI_DATASET_DIR, exist_ok=True)

    ui_tg_path = os.path.join(UI_DATASET_DIR, f"{normal_base}.TextGrid")
    with open(ui_tg_path, "w", encoding="utf-8") as f:
        f.write(tg_normal)
    
    # Update file_status.json to mark as verified
    file_status = init_file_status()
    if filename in file_status:
        file_status[filename]["verified"] = True
        file_status[filename]["verified_by"] = verified_by
        file_status[filename]["verified_at"] = datetime.now().isoformat()
        file_status[filename]["verification_file"] = verified_filename
        file_status[filename]["verification_tg_4x"] = tg_4x_filename
        file_status[filename]["verification_tg_normal"] = tg_normal_filename
        save_file_status(file_status)
    
    return jsonify({
        "success": True,
        "message": "File verified successfully",
        "verified_file": verified_filename,
        "tg_4x": tg_4x_filename,
        "tg_normal": tg_normal_filename
    })


# ==============================
# VERIFICATION AUTO-SAVE ROUTES
# ==============================

def get_verification_autosave_path(username, filename):
    """Get path for verification auto-save file"""
    user_autosave_dir = os.path.join(AUTOSAVE_FOLDER, "verification", username)
    os.makedirs(user_autosave_dir, exist_ok=True)
    
    base = os.path.splitext(filename)[0]
    autosave_filename = f"{base}_verification_autosave.json"
    return os.path.join(user_autosave_dir, autosave_filename)

@app.route('/api/verification-autosave', methods=['POST'])
def verification_autosave():
    """Auto-save verification progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    username = session["user"]
    audio_file = data.get("audio_file")
    annotator = data.get("annotator")
    frames = data.get("frames", [])
    edited_cells = data.get("edited_cells", [])
    
    if not audio_file or not annotator:
        return jsonify({"error": "missing data"}), 400
    
    autosave_path = get_verification_autosave_path(username, audio_file)
    
    autosave_data = {
        "audio_file": audio_file,
        "annotator": annotator,
        "verifier": username,
        "last_updated": datetime.now().isoformat(),
        "frames": frames,
        "edited_cells": edited_cells
    }
    
    with open(autosave_path, 'w', encoding='utf-8') as f:
        json.dump(autosave_data, f, indent=2, ensure_ascii=False)
    
    return jsonify({"message": "autosaved", "timestamp": datetime.now().isoformat()})

@app.route('/api/verification-autosave/<annotator>/<filename>', methods=['GET'])
def get_verification_autosave(annotator, filename):
    """Get auto-saved verification progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_verification_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        with open(autosave_path, 'r', encoding='utf-8') as f:
            autosave_data = json.load(f)
        return jsonify(autosave_data)
    
    return jsonify({"frames": [], "edited_cells": []})

@app.route('/api/verification-autosave/clear/<annotator>/<filename>', methods=['POST'])
def clear_verification_autosave(annotator, filename):
    """Clear auto-saved verification progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_verification_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    
    return jsonify({"message": "autosave cleared"})

# Initialize file status on startup
init_file_status()
load_akshar_tracking()
load_duration_tracking()

if __name__ == '__main__':
    print("=" * 50)
    print("WAV Annotation Server - Distributed Mode with Categories")
    print("=" * 50)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Available categories: {get_available_categories()}")
    print(f"Annotations folder: {ANNOTATIONS_FOLDER}")
    print(f"Autosave folder: {AUTOSAVE_FOLDER}")
    print(f"Recordings folder: {RECORDINGS_FOLDER}")
    print(f"Users: {USERS_FILE}")
    print(f"File status: {FILE_STATUS_FILE}")
    print(f"Akshar tracking: {AKSHAR_TRACKING_FILE}")
    print(f"Duration tracking: {DURATION_TRACKING_FILE}")
    print(f"Daily target: {AKSHAR_DAILY_TARGET} akshars")
    print(f"Overall target: {AKSHAR_OVERALL_TARGET} akshars")
    print("=" * 50)
    
    # Run with SocketIO instead of app.run()
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)