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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

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



import shutil
from pathlib import Path


# Add these helper functions near the top of your app.py (after imports)

# ==============================
# TEXTGRID TIER GENERATION HELPERS
# ==============================

AKSHAR_SET = {
    "अ", "आ", "इ", "उ", "ए", "ओ",
    "क","ख","ग","घ","च","छ","ज","झ",
    "ट","ठ","ड","ढ","त","थ","द","ध",
    "प","फ","ब","भ",
    "न","म",
    "य","र","ل","व",
    "स","ह",
    "ं","ँ","ॉ","०"
}

VYANJAN_SET = {
    "क","ख","ग","घ","च","छ","ज","झ",
    "ट","ठ","ड","ढ","त","थ","द","ध",
    "प","फ","ब","भ",
    "न","म","य","र","ل","व","स","ह"
}

SWAR_SET = {"अ","आ","इ","ई","उ","ऊ","ए","ऐ","ओ","औ"}
NAASIKA_SET = {"म","न","ं","ँ"}

# Normalization map for cleaning
norm_map = {
    "ा": "आ", "ि": "इ", "ी": "इ", "ु": "उ", "ू": "उ",
    "े": "ए", "ै": "ए", "ो": "ओ", "ौ": "ओ", "ृ": "ऋ",
    "ँ": "ं",
    "ण": "न", "ङ": "न", "ञ": "न",
    "श": "स", "ष": "स",
    "ई": "इ", "ऊ": "उ", "ऐ": "ए", "औ": "ओ"
}

def merge_akshars(a, b):
    """Merge two akshar strings"""
    a, b = a.strip(), b.strip()
    
    if not a: return b
    if not b: return a
    if a == b: return a
    if b.startswith(a): return b
    if a.endswith(b): return a
    
    for i in range(len(a)):
        if b.startswith(a[i:]):
            return a[:i] + b
    
    return a + b

def clean_text(text):
    """Clean and normalize text"""
    text = text.strip()
    if text == "":
        return ""
    
    # normalize + filter
    chars = []
    for ch in text:
        if ch in norm_map:
            ch = norm_map[ch]
        if ch in AKSHAR_SET:
            chars.append(ch)
    
    if not chars:
        return ""
    
    # dedup
    dedup = [chars[0]]
    for ch in chars[1:]:
        if ch != dedup[-1]:
            dedup.append(ch)
    
    # remove implicit अ
    final = []
    i = 0
    while i < len(dedup):
        ch = dedup[i]
        if (ch in VYANJAN_SET and
            i + 1 < len(dedup) and
            dedup[i + 1] == "अ"):
            final.append(ch)
            i += 2
        else:
            final.append(ch)
            i += 1
    
    return "".join(final)

def get_swar(text):
    """Extract swar (vowels) from text"""
    text = text.strip()
    if text == "":
        return ""
    
    swars = []
    i = 0
    while i < len(text):
        ch = text[i]
        
        if ch in SWAR_SET:
            swars.append(ch)
        
        elif ch in VYANJAN_SET:
            if i + 1 < len(text) and text[i + 1] in SWAR_SET:
                swars.append(text[i + 1])
                i += 1
            else:
                swars.append("अ")
        
        i += 1
    
    return "".join(swars)

def get_vyanjan(text):
    """Extract vyanjan (consonants) from text"""
    out = []
    for ch in text:
        if ch in VYANJAN_SET:
            if not out or out[-1] != ch:
                out.append(ch)
    return "".join(out)

def get_naasika(text):
    """Extract naasika (nasal sounds) from text"""
    out = []
    for ch in text:
        if ch in NAASIKA_SET:
            if not out or out[-1] != ch:
                out.append(ch)
    return "".join(out)



def create_enhanced_textgrid(frames, duration, sentence, annotator, window_ms=54):
    """
    Create TextGrid with multiple tiers:
    - sentence: original sentence
    - annotations: original windows (54ms for normal, 216ms for 4x)
    - window_108ms: merged 108ms windows (merges every 2 frames)
    - swar: extracted vowels from merged frames
    - vyanjan: extracted consonants from merged frames
    - naasika: extracted nasal sounds from merged frames
    - annotator: annotator name
    
    For 4x files: frames are 216ms, scaled to 54ms, then merged to 108ms
    For normal files: frames are already 108ms, so just clean them
    """
    tg = []
    
    tg.append('File type = "ooTextFile"')
    tg.append('Object class = "TextGrid"\n')
    
    tg.append(f"xmin = 0")
    tg.append(f"xmax = {duration}")
    tg.append("tiers? <exists>")
    tg.append("size = 7")
    tg.append("item []:")
    
    # ========== 1. sentence tier ==========
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
    
    # ========== 2. annotations tier (original windows) ==========
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
    
    # ========== 3. window_108ms tier (merge every 2 frames) ==========
    # Determine if we need to merge based on window size
    # If frames are 54ms, merge pairs to get 108ms
    # If frames are already 108ms, just clean them
    
    merged_frames = []
    
    # Check the duration of first frame to determine if merging is needed
    if len(frames) > 0:
        first_frame_duration = frames[0]["end_ms"] - frames[0]["start_ms"]
        print(f"First frame duration: {first_frame_duration}ms")
        
        if first_frame_duration == 54:
            # Need to merge 54ms frames to 108ms
            print(f"Merging 54ms frames to 108ms. Total frames: {len(frames)}")
            
            i = 0
            while i < len(frames):
                frame1 = frames[i]
                text1 = frame1["text"] if frame1["text"] else ""
                
                if i + 1 < len(frames):
                    frame2 = frames[i + 1]
                    text2 = frame2["text"] if frame2["text"] else ""
                    
                    # Merge the two texts
                    if text1 and text2:
                        merged_text = merge_akshars(text1, text2)
                    elif text1:
                        merged_text = text1
                    elif text2:
                        merged_text = text2
                    else:
                        merged_text = ""
                    
                    # Create 108ms window from the pair
                    start_ms = frame1["start_ms"]
                    end_ms = frame2["end_ms"]
                    
                    i += 2  # Move to next pair
                else:
                    # Odd number of frames - keep as 54ms
                    merged_text = text1
                    start_ms = frame1["start_ms"]
                    end_ms = frame1["end_ms"]
                    i += 1
                
                # Clean the text
                cleaned_text = clean_text(merged_text) if merged_text else ""
                
                merged_frames.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": cleaned_text
                })
            
            print(f"Merged {len(frames)} frames into {len(merged_frames)} frames")
            
        else:
            # Frames are already 108ms or other size, just clean them
            print(f"Frames are already {first_frame_duration}ms, just cleaning text")
            for f in frames:
                cleaned_text = clean_text(f["text"]) if f["text"] else ""
                merged_frames.append({
                    "start_ms": f["start_ms"],
                    "end_ms": f["end_ms"],
                    "text": cleaned_text
                })
    else:
        print("No frames to process")
    
    tg.append("    item [3]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "window_108ms"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(merged_frames)}")
    
    for i, f in enumerate(merged_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = f["text"] if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 4. swar tier (from merged frames) ==========
    tg.append("    item [4]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "swar"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(merged_frames)}")
    
    for i, f in enumerate(merged_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = get_swar(f["text"]) if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 5. vyanjan tier (from merged frames) ==========
    tg.append("    item [5]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "vyanjan"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(merged_frames)}")
    
    for i, f in enumerate(merged_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = get_vyanjan(f["text"]) if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 6. naasika tier (from merged frames) ==========
    tg.append("    item [6]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "naasika"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(merged_frames)}")
    
    for i, f in enumerate(merged_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = get_naasika(f["text"]) if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 7. annotator tier ==========
    tg.append("    item [7]:")
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

def create_enhanced_normal_textgrid(frames, duration, sentence, annotator, window_ms=108):
    """
    Create TextGrid for normal files (already at 108ms windows)
    - annotations tier: original frames (already 108ms)
    - window_108ms tier: same as annotations but with cleaned text
    - swar, vyanjan, naasika tiers: derived from cleaned text
    """
    tg = []
    
    tg.append('File type = "ooTextFile"')
    tg.append('Object class = "TextGrid"\n')
    
    tg.append(f"xmin = 0")
    tg.append(f"xmax = {duration}")
    tg.append("tiers? <exists>")
    tg.append("size = 7")
    tg.append("item []:")
    
    # ========== 1. sentence tier ==========
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
    
    # ========== 2. annotations tier (original 108ms frames) ==========
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
    
    # ========== 3. window_108ms tier (cleaned version, same duration) ==========
    # For normal files, frames are already 108ms, so we just clean the text
    cleaned_frames = []
    for f in frames:
        cleaned_text = clean_text(f["text"]) if f["text"] else ""
        cleaned_frames.append({
            "start_ms": f["start_ms"],
            "end_ms": f["end_ms"],
            "text": cleaned_text
        })
    
    tg.append("    item [3]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "window_108ms"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(cleaned_frames)}")
    
    for i, f in enumerate(cleaned_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = f["text"] if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 4. swar tier (from cleaned text) ==========
    tg.append("    item [4]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "swar"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(cleaned_frames)}")
    
    for i, f in enumerate(cleaned_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = get_swar(f["text"]) if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 5. vyanjan tier (from cleaned text) ==========
    tg.append("    item [5]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "vyanjan"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(cleaned_frames)}")
    
    for i, f in enumerate(cleaned_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = get_vyanjan(f["text"]) if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 6. naasika tier (from cleaned text) ==========
    tg.append("    item [6]:")
    tg.append('        class = "IntervalTier"')
    tg.append('        name = "naasika"')
    tg.append(f"        xmin = 0")
    tg.append(f"        xmax = {duration}")
    tg.append(f"        intervals: size = {len(cleaned_frames)}")
    
    for i, f in enumerate(cleaned_frames, 1):
        start = f["start_ms"] / 1000.0
        end = f["end_ms"] / 1000.0
        text = get_naasika(f["text"]) if f["text"] else ""
        
        tg.append(f"        intervals [{i}]:")
        tg.append(f"            xmin = {start}")
        tg.append(f"            xmax = {end}")
        tg.append(f'            text = "{text}"')
    
    # ========== 7. annotator tier ==========
    tg.append("    item [7]:")
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

# ==============================
# DATE-WISE ORGANIZED SAVING (ADDITIONAL -不影响现有功能)
# ==============================

def get_date_folder():
    """Get current date folder name (YYYY-MM-DD)"""
    return get_current_ist_date()

def copy_file_with_date_org(source_path, dest_dir, filename):
    """Copy a file to date-wise organized folder"""
    try:
        if not os.path.exists(source_path):
            print(f"Source file not found: {source_path}")
            return False
        
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(source_path, dest_path)
        print(f"Copied to date folder: {dest_path}")
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

def save_to_date_wise_ui_dataset(audio_filename, username, frames, sentence, full_sequence, file_type='4x'):
    """
    Save complete package to UI_DATASET or NORMAL_UI_DATASET date-wise folder
    This is an ADDITIONAL save, does NOT replace existing saves
    """
    try:
        date_folder = get_date_folder()
        base = os.path.splitext(audio_filename)[0]
        
        # Determine destination folder and file paths
        if file_type == '4x':
            # For 4x files - save normal versions
            dest_base = os.path.join("DATE_WISE_DATA", date_folder, "UI_DATASET")
            normal_base = base.replace("_4x", "")
            
            # Source files
            json_source = os.path.join(ANNOTATIONS_FOLDER, username, f"{base}.json")
            audio_source = find_audio_file(normal_base + ".wav")
            tg_source = os.path.join(ANNOTATIONS_FOLDER, username, f"{normal_base}.TextGrid")
            
            # Destination filenames (use normal base name)
            dest_json = f"{normal_base}.json"
            dest_audio = f"{normal_base}.wav"
            dest_tg = f"{normal_base}.TextGrid"
        else:
            # For normal files - save as-is
            dest_base = os.path.join("DATE_WISE_DATA", date_folder, "NORMAL_UI_DATASET")
            normal_base = base
            
            # Source files
            json_source = os.path.join(NORMAL_ANNOTATIONS_FOLDER, username, f"{base}.json")
            audio_source = find_normal_audio_file(audio_filename)
            tg_source = os.path.join(NORMAL_ANNOTATIONS_FOLDER, username, f"{base}.TextGrid")
            
            # Destination filenames (use same base name)
            dest_json = f"{base}.json"
            dest_audio = f"{base}.wav"
            dest_tg = f"{base}.TextGrid"
        
        # Create destination directory
        os.makedirs(dest_base, exist_ok=True)
        
        # Copy JSON file
        if os.path.exists(json_source):
            copy_file_with_date_org(json_source, dest_base, dest_json)
        else:
            # Create JSON if it doesn't exist (for UI_DATASET)
            json_data = {
                "audio_file": audio_filename,
                "annotator": username,
                "timestamp": datetime.now().isoformat(),
                "sentence": sentence,
                "full_sequence": full_sequence,
                "frames": frames,
                "note": "Auto-created for UI_DATASET"
            }
            dest_json_path = os.path.join(dest_base, dest_json)
            with open(dest_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"Created JSON for UI_DATASET: {dest_json_path}")
        
        # Copy audio file
        if audio_source and os.path.exists(audio_source):
            copy_file_with_date_org(audio_source, dest_base, dest_audio)
        else:
            print(f"Warning: Audio file not found for {normal_base}")
        
        # Copy TextGrid file
        if os.path.exists(tg_source):
            copy_file_with_date_org(tg_source, dest_base, dest_tg)
        else:
            print(f"Warning: TextGrid not found for {normal_base}")
        
        print(f"Date-wise UI_DATASET save complete for {normal_base}")
        return True
        
    except Exception as e:
        print(f"Error in date-wise UI_DATASET save: {e}")
        return False

def save_to_date_wise_verified(original_filename, original_annotator, verified_by, frames, sentence, full_sequence, file_type='4x'):
    """
    Save complete package to verified or normal_verified date-wise folder
    This is an ADDITIONAL save, does NOT replace existing saves
    """
    try:
        date_folder = get_date_folder()
        base = os.path.splitext(original_filename)[0]
        
        if file_type == '4x':
            # For 4x files - save normal versions
            dest_base = os.path.join("DATE_WISE_DATA", date_folder, "verified")
            normal_base = base.replace("_4x", "")
            
            # Source files
            json_source = os.path.join("verified", f"{base}.json")
            audio_source = find_audio_file(normal_base + ".wav")
            tg_source = os.path.join("verified", f"{normal_base}.TextGrid")
            
            # Destination filenames (use normal base name)
            dest_json = f"{normal_base}.json"
            dest_audio = f"{normal_base}.wav"
            dest_tg = f"{normal_base}.TextGrid"
        else:
            # For normal files - save as-is
            dest_base = os.path.join("DATE_WISE_DATA", date_folder, "normal_verified")
            normal_base = base
            
            # Source files
            json_source = os.path.join("normal_verified", f"{base}.json")
            audio_source = find_normal_audio_file(original_filename)
            tg_source = os.path.join("normal_verified", f"{base}.TextGrid")
            
            # Destination filenames (use same base name)
            dest_json = f"{base}.json"
            dest_audio = f"{base}.wav"
            dest_tg = f"{base}.TextGrid"
        
        # Create destination directory
        os.makedirs(dest_base, exist_ok=True)
        
        # Copy JSON file
        if os.path.exists(json_source):
            copy_file_with_date_org(json_source, dest_base, dest_json)
        else:
            # Create JSON if it doesn't exist
            json_data = {
                "audio_file": original_filename,
                "original_annotator": original_annotator,
                "verified_by": verified_by,
                "verified_at": datetime.now().isoformat(),
                "sentence": sentence,
                "full_sequence": full_sequence,
                "frames": frames
            }
            dest_json_path = os.path.join(dest_base, dest_json)
            with open(dest_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"Created JSON for verified: {dest_json_path}")
        
        # Copy audio file
        if audio_source and os.path.exists(audio_source):
            copy_file_with_date_org(audio_source, dest_base, dest_audio)
        else:
            print(f"Warning: Audio file not found for {normal_base}")
        
        # Copy TextGrid file
        if os.path.exists(tg_source):
            copy_file_with_date_org(tg_source, dest_base, dest_tg)
        else:
            print(f"Warning: TextGrid not found for {normal_base}")
        
        print(f"Date-wise verified save complete for {normal_base}")
        return True
        
    except Exception as e:
        print(f"Error in date-wise verified save: {e}")
        return False


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
                "category": category,
                "verified": False,  # Add this line
                "verified_by": None,  # Add this to track who verified
                "verified_at": None   # Add this to track when verified
            }
        else:
            # Ensure category exists for existing files
            if "category" not in updated_status[filename]:
                updated_status[filename]["category"] = category
            # Ensure priority flag exists
            if "priority" not in updated_status[filename]:
                updated_status[filename]["priority"] = 1 if filename.startswith('BEEJ_') else 0
            # Ensure verified field exists for existing files
            if "verified" not in updated_status[filename]:
                updated_status[filename]["verified"] = False
            if "verified_by" not in updated_status[filename]:
                updated_status[filename]["verified_by"] = None
            if "verified_at" not in updated_status[filename]:
                updated_status[filename]["verified_at"] = None

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
    # 🔥 ENHANCED TEXTGRID GENERATION (with multiple tiers)
    # =========================
    
    # 🔹 4x TG with enhanced tiers (frames are 216ms from user)
    # The 4x TextGrid should have annotations tier with 216ms windows
    duration_4x = frames[-1]["end_ms"] / 1000.0 if frames else 0
    tg_4x = create_enhanced_textgrid(
        frames,  # These are 216ms frames from the user (4x slowed)
        duration_4x, 
        data.get("full_sequence", ""), 
        username,
        window_ms=216  # 4x window is 216ms
    )
    
    tg_4x_path = os.path.join(user_dir, f"{base}.TextGrid")
    with open(tg_4x_path, "w", encoding="utf-8") as f:
        f.write(tg_4x)
    
    # 🔹 NORMAL TG - Scale frames from 216ms to 54ms
    def scale_frames(frames, factor):
        return [
            {
                "start_ms": f["start_ms"] / factor,
                "end_ms": f["end_ms"] / factor,
                "text": f["text"]
            }
            for f in frames
        ]
    
    # Scale 216ms frames to 54ms frames
    normal_frames = scale_frames(frames, 4)
    normal_duration = normal_frames[-1]["end_ms"] / 1000.0 if normal_frames else 0
    
    normal_base = base.replace("_4x", "")
    
    # IMPORTANT: For normal TextGrid from 4x files, use create_enhanced_textgrid 
    # (which merges 54ms frames to 108ms windows), NOT create_enhanced_normal_textgrid
    tg_normal = create_enhanced_textgrid(
        normal_frames,  # These are 54ms frames (scaled from 4x)
        normal_duration, 
        data.get("full_sequence", ""), 
        username,
        window_ms=54  # Normal window is 54ms, will be merged to 108ms
    )
    
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
    
    # =========================
    # 🔥 NEW: DATE-WISE ORGANIZED SAVING
    # =========================
    try:
        save_to_date_wise_ui_dataset(
            audio_filename=audio_file,
            username=username,
            frames=frames,
            sentence=data.get("sentence", ""),
            full_sequence=data.get("full_sequence", ""),
            file_type='4x'
        )
    except Exception as e:
        print(f"Warning: Date-wise UI_DATASET save failed: {e}")
    
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
        
        # Combine both 4x and normal completed files
        completed_files = len(get_user_completed_files(username)) + len(get_normal_user_completed_files(username))
        
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
    
    # Save verified annotation
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
    # 🔥 ENHANCED TEXTGRID GENERATION FOR VERIFIED FILES
    # =========================
    
    # Get audio file duration for 4x version
    audio_path = find_audio_file(filename)
    if audio_path and os.path.exists(audio_path):
        info = sf.info(audio_path)
        duration_4x = info.duration
    else:
        duration_4x = frames[-1]["end_ms"] / 1000.0 if frames else 0
    
    # 🔹 Generate 4x TextGrid with enhanced tiers
    tg_4x = create_enhanced_textgrid(
        frames, 
        duration_4x, 
        data.get("full_sequence", ""), 
        username,
        window_ms=216
    )
    
    # Save 4x TextGrid in verified folder
    tg_4x_filename = f"{base}.TextGrid"
    tg_4x_path = os.path.join(VERIFIED_FOLDER, tg_4x_filename)
    with open(tg_4x_path, "w", encoding="utf-8") as f:
        f.write(tg_4x)
    
    # 🔹 Generate Normal TextGrid (scale frames by factor 4)
    def scale_frames(frames, factor):
        return [
            {
                "start_ms": f["start_ms"] / factor,
                "end_ms": f["end_ms"] / factor,
                "text": f["text"]
            }
            for f in frames
        ]
    
    normal_frames = scale_frames(frames, 4)
    normal_base = base.replace("_4x", "")
    
    # Get normal audio file duration
    normal_audio_path = find_audio_file(normal_base + ".wav")
    if normal_audio_path and os.path.exists(normal_audio_path):
        info = sf.info(normal_audio_path)
        normal_duration = info.duration
    else:
        normal_duration = normal_frames[-1]["end_ms"] / 1000.0 if normal_frames else 0
    
    # Use create_enhanced_textgrid for normal version (merges 54ms to 108ms)
    tg_normal = create_enhanced_textgrid(
        normal_frames, 
        normal_duration, 
        data.get("full_sequence", ""), 
        username,
        window_ms=54
    )
    
    # Save normal TextGrid in verified folder
    tg_normal_filename = f"{normal_base}.TextGrid"
    tg_normal_path = os.path.join(VERIFIED_FOLDER, tg_normal_filename)
    with open(tg_normal_path, "w", encoding="utf-8") as f:
        f.write(tg_normal)
    
    # =========================
    # 🔥 SAVE TO UI_DATASET (normal version)
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
    
    # =========================
    # 🔥 NEW: DATE-WISE ORGANIZED SAVING
    # =========================
    try:
        save_to_date_wise_verified(
            original_filename=filename,
            original_annotator=username,
            verified_by=verified_by,
            frames=frames,
            sentence=data.get("sentence", ""),
            full_sequence=data.get("full_sequence", ""),
            file_type='4x'
        )
    except Exception as e:
        print(f"Warning: Date-wise verified save failed: {e}")
    
    return jsonify({
        "success": True,
        "message": "File verified successfully",
        "verified_file": verified_filename,
        "tg_4x": tg_4x_filename,
        "tg_normal": tg_normal_filename
    })

def reverse_user_stats(username, akshar_count, duration_seconds):
    """Reverse (subtract) user's stats when an annotation is rejected"""
    
    # Reverse akshar counts
    akshar_tracking = load_akshar_tracking()
    current_date = get_current_ist_date()
    
    if current_date in akshar_tracking["daily"]:
        if username in akshar_tracking["daily"][current_date]:
            akshar_tracking["daily"][current_date][username] = max(0, akshar_tracking["daily"][current_date][username] - akshar_count)
    
    if username in akshar_tracking["overall"]:
        akshar_tracking["overall"][username] = max(0, akshar_tracking["overall"][username] - akshar_count)
    
    save_akshar_tracking(akshar_tracking)
    
    # Reverse duration counts
    duration_tracking = load_duration_tracking()
    
    if current_date in duration_tracking["daily"]:
        if username in duration_tracking["daily"][current_date]:
            duration_tracking["daily"][current_date][username] = max(0, duration_tracking["daily"][current_date][username] - duration_seconds)
    
    if username in duration_tracking["overall"]:
        duration_tracking["overall"][username] = max(0, duration_tracking["overall"][username] - duration_seconds)
    
    save_duration_tracking(duration_tracking)
    
    print(f"Reversed stats for {username}: -{akshar_count} akshars, -{duration_seconds}s")


@app.route('/api/verify-reject', methods=['POST'])
def verify_reject():
    """Reject a completed annotation and mark it as pending for re-annotation"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    print(f"Received reject request data: {data}")
    
    filename = data.get("filename")
    annotator = data.get("annotator")
    rejected_by = data.get("rejected_by", session["user"])
    
    if not filename or not annotator:
        return jsonify({"error": "Missing data: filename and annotator are required"}), 400
    
    file_status = init_file_status()
    
    # Check if file exists in status
    if filename not in file_status:
        return jsonify({"error": "File not found in status"}), 404
    
    # Load the annotation to get frames and duration before deleting
    annotation_file = file_status[filename].get("annotation_file")
    frames = []
    duration_seconds = 0
    
    if annotation_file:
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotator, annotation_file)
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                    frames = annotation_data.get("frames", [])
                
                # Get audio file duration
                audio_path = find_audio_file(filename)
                if audio_path and os.path.exists(audio_path):
                    info = sf.info(audio_path)
                    duration_seconds = info.duration
                    
            except Exception as e:
                print(f"Error reading annotation file: {e}")
    
    # Calculate akshar count from frames
    akshar_count = sum(1 for frame in frames if frame.get("text") and frame["text"].strip() != "")
    
    print(f"Rejecting file - Akshar count: {akshar_count}, Duration: {duration_seconds}s")
    
    # Reverse the user's stats (subtract from their counts)
    if akshar_count > 0 or duration_seconds > 0:
        reverse_user_stats(annotator, akshar_count, duration_seconds)
        print(f"Reversed stats for {annotator}: -{akshar_count} akshars, -{duration_seconds}s")
    
    # Delete or move the annotation files
    REJECTED_FOLDER = "rejected_annotations"
    os.makedirs(REJECTED_FOLDER, exist_ok=True)
    
    deleted_files = []
    
    # Move annotation JSON file instead of deleting (for audit trail)
    if annotation_file:
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotator, annotation_file)
        if os.path.exists(annotation_path):
            rejected_path = os.path.join(REJECTED_FOLDER, f"{annotator}_{annotation_file}")
            os.rename(annotation_path, rejected_path)
            deleted_files.append(f"Moved: {annotation_file} -> rejected/")
            print(f"Moved annotation to: {rejected_path}")
        
        # Also move TextGrid files
        base = os.path.splitext(filename)[0]
        tg_4x_path = os.path.join(ANNOTATIONS_FOLDER, annotator, f"{base}.TextGrid")
        if os.path.exists(tg_4x_path):
            rejected_tg_path = os.path.join(REJECTED_FOLDER, f"{annotator}_{base}.TextGrid")
            os.rename(tg_4x_path, rejected_tg_path)
            deleted_files.append(f"Moved: {base}.TextGrid -> rejected/")
        
        normal_base = base.replace("_4x", "")
        tg_normal_path = os.path.join(ANNOTATIONS_FOLDER, annotator, f"{normal_base}.TextGrid")
        if os.path.exists(tg_normal_path):
            rejected_normal_path = os.path.join(REJECTED_FOLDER, f"{annotator}_{normal_base}.TextGrid")
            os.rename(tg_normal_path, rejected_normal_path)
            deleted_files.append(f"Moved: {normal_base}.TextGrid -> rejected/")
    
    # Clean up verification files if they exist
    VERIFIED_FOLDER = "verified"
    base = os.path.splitext(filename)[0]
    
    verified_json = os.path.join(VERIFIED_FOLDER, f"{base}.json")
    verified_tg_4x = os.path.join(VERIFIED_FOLDER, f"{base}.TextGrid")
    normal_base = base.replace("_4x", "")
    verified_tg_normal = os.path.join(VERIFIED_FOLDER, f"{normal_base}.TextGrid")
    
    for file_path in [verified_json, verified_tg_4x, verified_tg_normal]:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_files.append(f"Deleted: {os.path.basename(file_path)}")
    
    # Remove from UI_DATASET
    UI_DATASET_DIR = "UI_DATASET"
    ui_tg_path = os.path.join(UI_DATASET_DIR, f"{normal_base}.TextGrid")
    if os.path.exists(ui_tg_path):
        os.remove(ui_tg_path)
        deleted_files.append(f"Deleted from UI_DATASET: {normal_base}.TextGrid")
    
    # Update file_status.json - mark as pending for re-annotation
    file_status[filename]["status"] = "pending"
    file_status[filename]["assigned_to"] = None
    file_status[filename]["assigned_at"] = None
    file_status[filename]["completed_at"] = None
    file_status[filename]["annotation_file"] = None
    file_status[filename]["verified"] = False
    file_status[filename]["verified_by"] = None
    file_status[filename]["verified_at"] = None
    file_status[filename]["verification_file"] = None
    file_status[filename]["verification_tg_4x"] = None
    file_status[filename]["verification_tg_normal"] = None
    
    save_file_status(file_status)
    
    # Clear autosave files
    verification_autosave_path = get_verification_autosave_path(rejected_by, filename)
    if os.path.exists(verification_autosave_path):
        os.remove(verification_autosave_path)
    
    regular_autosave_path = get_autosave_path(annotator, filename)
    if os.path.exists(regular_autosave_path):
        os.remove(regular_autosave_path)
    
    # Log the rejection
    REJECTION_LOG_FOLDER = "rejection_logs"
    os.makedirs(REJECTION_LOG_FOLDER, exist_ok=True)
    rejection_log = os.path.join(REJECTION_LOG_FOLDER, f"{base}_rejection.json")
    
    rejection_data = {
        "filename": filename,
        "original_annotator": annotator,
        "rejected_by": rejected_by,
        "rejected_at": datetime.now().isoformat(),
        "reason": data.get("reason", "Annotation was incorrect"),
        "akshar_count_removed": akshar_count,
        "duration_seconds_removed": duration_seconds,
        "deleted_annotation_file": annotation_file,
        "deleted_files": deleted_files
    }
    
    with open(rejection_log, 'w', encoding='utf-8') as f:
        json.dump(rejection_data, f, indent=2, ensure_ascii=False)
    
    return jsonify({
        "success": True,
        "message": f"File rejected. Removed {akshar_count} akshars and {duration_seconds:.1f}s from {annotator}'s stats",
        "akshar_removed": akshar_count,
        "duration_removed": duration_seconds,
        "deleted_files": deleted_files,
        "rejection_log": os.path.basename(rejection_log)
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


def mark_file_verified(filename, verified_by, is_normal=False):
    """Mark a file as verified"""
    if is_normal:
        file_status = init_normal_file_status()
        save_func = save_normal_file_status
    else:
        file_status = init_file_status()
        save_func = save_file_status
    
    if filename in file_status:
        file_status[filename]["verified"] = True
        file_status[filename]["verified_by"] = verified_by
        file_status[filename]["verified_at"] = datetime.now().isoformat()
        save_func(file_status)
        return True
    
    return False

def get_user_total_completed_files(username):
    """Get total completed files (both 4x and normal) for a user"""
    # 4x files
    _4x_count = len(get_user_completed_files(username))
    
    # Normal files
    normal_count = len(get_normal_user_completed_files(username))
    
    return _4x_count + normal_count

def get_user_total_duration(username):
    """Get total duration from both 4x and normal annotations"""
    tracking = load_duration_tracking()
    return tracking["overall"].get(username, 0)



# Add these constants at the top with other constants
NORMAL_DATA_FOLDER = 'normal_data'  # New folder for normal WAV files
NORMAL_ANNOTATIONS_FOLDER = "normal_annotations"  # For normal file annotations
NORMAL_AUTOSAVE_FOLDER = "normal_autosave"  # For normal file autosaves
NORMAL_FILE_STATUS_FILE = "normal_file_status.json"  # Tracking for normal files
NORMAL_UI_DATASET_DIR = "NORMAL_UI_DATASET"  # For normal file TextGrids

# Create necessary directories
os.makedirs(NORMAL_DATA_FOLDER, exist_ok=True)
os.makedirs(NORMAL_ANNOTATIONS_FOLDER, exist_ok=True)
os.makedirs(NORMAL_AUTOSAVE_FOLDER, exist_ok=True)
os.makedirs(NORMAL_UI_DATASET_DIR, exist_ok=True)

# Add WINDOW_NORMAL constant
WINDOW_NORMAL = 0.108  # 108ms windows for normal files

# ==============================
# NORMAL FILE MANAGEMENT
# ==============================

def get_normal_files_by_category(category=None):
    """
    Get all .wav files (not _4x) from normal_data folder
    Returns list of tuples (filename, category)
    """
    files = []
    
    if category is None or category == 'all':
        # Get all files from all subfolders
        for root, dirs, files_in_dir in os.walk(NORMAL_DATA_FOLDER):
            for file in files_in_dir:
                if file.endswith('.wav') and '_4x' not in file:
                    # Get relative path to determine category
                    rel_path = os.path.relpath(root, NORMAL_DATA_FOLDER)
                    if rel_path == '.':
                        # Files directly in normal_data folder
                        file_category = 'root'
                    else:
                        # Get the top-level folder name as category
                        # For normal_data/beej_mantra/file.wav -> 'beej_mantra'
                        # For normal_data/beej_mantra/subfolder/file.wav -> 'beej_mantra'
                        file_category = rel_path.split(os.sep)[0]
                    files.append((file, file_category))
                    print(f"Found normal file: {file} in category: {file_category}")  # Debug print
    else:
        # Get files only from specific category folder
        category_path = os.path.join(NORMAL_DATA_FOLDER, category)
        if os.path.exists(category_path):
            for root, dirs, files_in_dir in os.walk(category_path):
                for file in files_in_dir:
                    if file.endswith('.wav') and '_4x' not in file:
                        files.append((file, category))
                        print(f"Found normal file in {category}: {file}")  # Debug print
    
    return files

def init_normal_file_status():
    """Initialize normal file status tracking if it doesn't exist"""
    all_files_with_categories = get_normal_files_by_category()
    
    # Load existing file status if it exists
    existing_status = {}
    if os.path.exists(NORMAL_FILE_STATUS_FILE):
        try:
            with open(NORMAL_FILE_STATUS_FILE, 'r', encoding='utf-8') as f:
                existing_status = json.load(f)
        except Exception as e:
            print("Error reading existing normal file status:", e)
    
    new_files = []
    updated_status = existing_status.copy()
    
    for filename, category in all_files_with_categories:
        if filename not in updated_status:
            print(f"Adding new normal file: {filename} (category: {category})")
            new_files.append(filename)
            updated_status[filename] = {
                "status": "pending",
                "assigned_to": None,
                "assigned_at": None,
                "completed_at": None,
                "annotation_file": None,
                "priority": 0,  # Set to 0 for all normal files
                "category": category,
                "verified": False,
                "verified_by": None,
                "verified_at": None
            }
        else:
            # Ensure status is pending if not completed and not assigned
            if (updated_status[filename].get("status") != "completed" and 
                updated_status[filename].get("assigned_to") is None):
                updated_status[filename]["status"] = "pending"
            
            # Update category for existing files if it changed
            if updated_status[filename].get("category") != category:
                print(f"Updating category for {filename} from {updated_status[filename].get('category')} to {category}")
                updated_status[filename]["category"] = category
            
            # Ensure priority is 0 for all normal files
            updated_status[filename]["priority"] = 0
            
            # Ensure verified field exists for existing files
            if "verified" not in updated_status[filename]:
                updated_status[filename]["verified"] = False
            if "verified_by" not in updated_status[filename]:
                updated_status[filename]["verified_by"] = None
            if "verified_at" not in updated_status[filename]:
                updated_status[filename]["verified_at"] = None
    
    if new_files:
        print(f"Added {len(new_files)} new normal files to tracking")
        save_normal_file_status(updated_status)
    elif updated_status != existing_status:
        print("Updating existing normal file status")
        save_normal_file_status(updated_status)
    
    return updated_status


def save_normal_file_status(file_status):
    """Save normal file status to JSON"""
    with open(NORMAL_FILE_STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(file_status, f, indent=2, ensure_ascii=False)

def get_normal_user_annotation_dir(username):
    """Get or create user's normal annotation directory"""
    user_dir = os.path.join(NORMAL_ANNOTATIONS_FOLDER, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_normal_user_completed_files(username):
    """Get list of normal files completed by a user from file_status.json only"""
    file_status = init_normal_file_status()
    completed_files = []
    
    for filename, status in file_status.items():
        if (status.get("status") == "completed" and 
            status.get("assigned_to") == username):
            completed_files.append(filename)
    
    return completed_files

def get_normal_next_file_for_user(username, category=None):
    """
    Get next unassigned or pending normal file for user
    If user already has an assigned file, return that file
    """
    file_status = init_normal_file_status()
    user_completed = get_normal_user_completed_files(username)
    skipped_files = user_skips.get(f"normal_{username}", [])
    
    print(f"\n=== DEBUG get_normal_next_file_for_user ===")
    print(f"Username: {username}")
    print(f"Category filter: {category}")
    print(f"Skipped files: {skipped_files}")
    print(f"Completed files (from file_status): {user_completed}")
    
    # FIRST: If user already has an assigned file (not completed)
    for filename, status in file_status.items():
        if (status.get("assigned_to") == username and 
            status.get("status") == "assigned"):
            # Check if file matches category filter
            if category is None or category == 'all' or status.get("category") == category:
                if filename not in user_completed:
                    print(f"Returning already assigned file: {filename}")
                    return filename
    
    # SECOND: Get all pending files (regardless of priority)
    pending_files = []
    for filename, status in file_status.items():
        # Check category filter
        category_match = False
        if category is None or category == 'all':
            category_match = True
        else:
            category_match = (status.get("category") == category)
        
        # Only include pending files not completed and not assigned to anyone
        if (status.get("status") == "pending" and 
            filename not in user_completed and
            status.get("assigned_to") is None and
            category_match):
            pending_files.append((filename, status))
    
    print(f"Pending files after filtering: {[f[0] for f in pending_files]}")
    
    # Filter out skipped files
    available_files = []
    for filename, status in pending_files:
        if filename not in skipped_files:
            available_files.append((filename, status))
        else:
            print(f"Skipping {filename} (in skip list)")
    
    print(f"Available files (not skipped): {[f[0] for f in available_files]}")
    
    # If no available files and we have skipped files, reset skips and try again
    if not available_files and skipped_files:
        print(f"No available files, resetting skip list for {username}")
        user_skips[f"normal_{username}"] = []
        
        # Now get all pending files without skip filter
        for filename, status in pending_files:
            available_files.append((filename, status))
        print(f"Available files after reset: {[f[0] for f in available_files]}")
    
    # Assign the first available file
    if available_files:
        available_files.sort(key=lambda x: x[0])  # Sort alphabetically
        filename, status = available_files[0]
        status["status"] = "assigned"
        status["assigned_to"] = username
        status["assigned_at"] = datetime.now().isoformat()
        save_normal_file_status(file_status)
        print(f"Assigned file: {filename}")
        return filename
    
    print("No files available!")
    return None

@app.route('/api/debug-normal-session')
def debug_normal_session():
    """Debug endpoint to check user session for normal files"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    
    return jsonify({
        "username": username,
        "normal_category_preference": get_user_normal_category_preference(username),
        "all_session_keys": list(session.keys()),
        "user_skips_normal": user_skips.get(f"normal_{username}", [])
    })


def mark_normal_file_completed(filename, username, annotation_filename):
    """Mark a normal file as completed by user"""
    file_status = init_normal_file_status()
    
    if filename in file_status:
        file_status[filename]["status"] = "completed"
        file_status[filename]["completed_at"] = datetime.now().isoformat()
        file_status[filename]["annotation_file"] = annotation_filename
        file_status[filename]["assigned_to"] = username
        save_normal_file_status(file_status)
        print(f"Marked {filename} as completed for {username}")
        return True
    
    return False

def get_normal_autosave_path(username, filename):
    """Get path for normal file auto-save"""
    user_autosave_dir = os.path.join(NORMAL_AUTOSAVE_FOLDER, username)
    os.makedirs(user_autosave_dir, exist_ok=True)
    
    base = os.path.splitext(filename)[0]
    autosave_filename = f"{base}_autosave.json"
    return os.path.join(user_autosave_dir, autosave_filename)

def find_normal_audio_file(filename):
    """Find full path of a normal file inside NORMAL_DATA_FOLDER recursively"""
    matches = glob.glob(os.path.join(NORMAL_DATA_FOLDER, '**', filename), recursive=True)
    return matches[0] if matches else None

# ==============================
# NORMAL FILE ANNOTATION ROUTES
# ==============================

@app.route('/api/fix-normal-completion-status', methods=['POST'])
def fix_normal_completion_status():
    """Fix: Mark files as completed in file_status.json based on actual annotations"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    file_status = init_normal_file_status()
    
    # Get actual annotation files from user's folder
    user_dir = get_normal_user_annotation_dir(username)
    annotated_files = []
    
    if os.path.exists(user_dir):
        for json_file in glob.glob(os.path.join(user_dir, "*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    audio_file = data.get("audio_file")
                    if audio_file:
                        annotated_files.append(audio_file)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    updated_files = []
    for filename in annotated_files:
        if filename in file_status:
            if file_status[filename].get("status") != "completed":
                file_status[filename]["status"] = "completed"
                file_status[filename]["completed_at"] = datetime.now().isoformat()
                file_status[filename]["annotation_file"] = f"{os.path.splitext(filename)[0]}.json"
                file_status[filename]["assigned_to"] = username
                updated_files.append(filename)
                print(f"Fixed: Marked {filename} as completed")
    
    if updated_files:
        save_normal_file_status(file_status)
    
    return jsonify({
        "message": "Completion status fixed",
        "annotated_files_found": annotated_files,
        "updated_in_status": updated_files
    })




@app.route('/normal-annotate')
def normal_annotate_page():
    """Normal file annotation page"""
    if not require_login():
        return redirect("/login")
    
    categories = get_normal_categories()
    return render_template("normal_annotate.html", user=session["user"], categories=categories)


def get_normal_categories():
    """Get all top-level subfolders in normal_data folder as categories"""
    categories = []
    try:
        for item in os.listdir(NORMAL_DATA_FOLDER):
            item_path = os.path.join(NORMAL_DATA_FOLDER, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                categories.append(item)
        # Always add 'root' if it exists or if there are files directly in normal_data
        if os.path.exists(os.path.join(NORMAL_DATA_FOLDER, 'root')) or any(f.endswith('.wav') for f in os.listdir(NORMAL_DATA_FOLDER) if os.path.isfile(os.path.join(NORMAL_DATA_FOLDER, f))):
            if 'root' not in categories:
                categories.append('root')
    except Exception as e:
        print(f"Error reading normal categories: {e}")
    
    # Always return at least ['all'] if no categories found
    if not categories:
        return ['all']
    return sorted(categories)


@app.route('/api/normal-categories')
def get_normal_categories_api():
    """Get available normal data categories"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    categories = get_normal_categories()
    return jsonify({
        "categories": categories,
        "current": get_user_normal_category_preference(session["user"])
    })

def get_user_normal_category_preference(username):
    """Get user's current normal category preference"""
    return session.get(f'normal_category_{username}', 'all')

def set_user_normal_category_preference(username, category):
    """Set user's normal category preference"""
    session[f'normal_category_{username}'] = category

@app.route('/api/normal-set-category', methods=['POST'])
def set_normal_category():
    """Set user's current normal category preference"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    category = data.get("category", "all")
    username = session["user"]
    
    # Validate category
    if category != "all" and category not in get_normal_categories():
        return jsonify({"error": "Invalid category"}), 400
    
    set_user_normal_category_preference(username, category)
    
    # Clear user's current assignment when switching categories
    file_status = init_normal_file_status()
    for filename, status in file_status.items():
        if status.get("assigned_to") == username and status.get("status") == "assigned":
            status["status"] = "pending"
            status["assigned_to"] = None
            status["assigned_at"] = None
    
    save_normal_file_status(file_status)
    
    return jsonify({
        "message": f"Switched to {category}",
        "category": category
    })

@app.route('/api/normal-next-file')
def get_normal_next_file():
    """Get the next normal file to annotate"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    category = get_user_normal_category_preference(username)
    next_file = get_normal_next_file_for_user(username, category)
    
    if next_file:
        return jsonify({
            "filename": next_file,
            "message": "File assigned successfully"
        })
    else:
        return jsonify({
            "filename": None,
            "message": f"No more normal files to annotate in {category}"
        })

@app.route('/api/normal-file-info/<filename>')
def get_normal_file_info(filename):
    """Get information about a specific normal file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    filepath = find_normal_audio_file(filename)
    
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

@app.route('/normal-audio/<filename>')
def serve_normal_audio(filename):
    """Serve normal audio file"""
    if not require_login():
        return redirect("/login")
    filepath = find_normal_audio_file(filename)
    if not filepath:
        return redirect("/login")
    
    directory = os.path.dirname(filepath)
    file_only = os.path.basename(filepath)
    
    return send_from_directory(directory, file_only)

@app.route('/api/normal-labels/<filename>')
def get_normal_labels(filename):
    """Get pre-existing labels for a normal file"""
    if not require_login():
        return jsonify({"error": "login required"}), 401
    
    base = os.path.splitext(filename)[0]
    filepath = find_normal_audio_file(filename)
    
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

@app.route('/api/normal-autosave', methods=['POST'])
def normal_autosave():
    """Auto-save normal file progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    username = session["user"]
    audio_file = data.get("audio_file")
    frames = data.get("frames", [])
    
    if not audio_file:
        return jsonify({"error": "no file specified"}), 400
    
    autosave_path = get_normal_autosave_path(username, audio_file)
    
    autosave_data = {
        "audio_file": audio_file,
        "annotator": username,
        "last_updated": datetime.now().isoformat(),
        "frames": frames
    }
    
    with open(autosave_path, 'w', encoding='utf-8') as f:
        json.dump(autosave_data, f, indent=2, ensure_ascii=False)
    
    return jsonify({"message": "autosaved", "timestamp": datetime.now().isoformat()})

@app.route('/api/normal-autosave/<filename>', methods=['GET'])
def get_normal_autosave(filename):
    """Get auto-saved progress for a normal file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_normal_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        with open(autosave_path, 'r', encoding='utf-8') as f:
            autosave_data = json.load(f)
        return jsonify(autosave_data)
    
    return jsonify({"frames": []})

@app.route('/api/normal-autosave/clear/<filename>', methods=['POST'])
def clear_normal_autosave(filename):
    """Clear auto-saved progress after successful submission"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_normal_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    
    return jsonify({"message": "autosave cleared"})


@app.route('/normal-submit', methods=['POST'])
def submit_normal_annotation():
    """Submit annotation for a normal file"""
    if not require_login():
        return jsonify({"error": "login required"}), 401
    
    data = request.json
    audio_file = data.get("audio_file")
    username = session["user"]
    frames = data.get("frames", [])
    category = get_user_normal_category_preference(username)
    
    # Get file duration
    filepath = find_normal_audio_file(audio_file)
    duration_seconds = 0
    if filepath and os.path.exists(filepath):
        info = sf.info(filepath)
        duration_seconds = info.duration
    
    # Update akshar counts
    akshar_update = update_akshar_counts(username, frames)
    
    # Update duration counts
    duration_update = update_duration_counts(username, duration_seconds)
    
    # Save JSON
    user_dir = get_normal_user_annotation_dir(username)
    base = os.path.splitext(audio_file)[0]
    annotation_filename = f"{base}.json"
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
    
    # Generate enhanced TextGrid for normal file
    tg_filename = f"{base}.TextGrid"
    tg_path = os.path.join(user_dir, tg_filename)
    
    tg_content = create_enhanced_normal_textgrid(
        frames, 
        duration_seconds, 
        data.get("full_sequence", ""), 
        username,
        window_ms=108
    )
    
    with open(tg_path, "w", encoding="utf-8") as f:
        f.write(tg_content)
    
    # Save to NORMAL_UI_DATASET
    ui_tg_path = os.path.join(NORMAL_UI_DATASET_DIR, f"{base}.TextGrid")
    with open(ui_tg_path, "w", encoding="utf-8") as f:
        f.write(tg_content)
    
    # Mark file as completed
    mark_normal_file_completed(audio_file, username, annotation_filename)
    
    # Clear autosave
    autosave_path = get_normal_autosave_path(username, audio_file)
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    
    # Get next file
    next_file = get_normal_next_file_for_user(username, category)
    
    # =========================
    # 🔥 NEW: DATE-WISE ORGANIZED SAVING (不影响现有功能)
    # =========================
    try:
        save_to_date_wise_ui_dataset(
            audio_filename=audio_file,
            username=username,
            frames=frames,
            sentence=data.get("sentence", ""),
            full_sequence=data.get("full_sequence", ""),
            file_type='normal'
        )
    except Exception as e:
        print(f"Warning: Date-wise NORMAL_UI_DATASET save failed (不影响主流程): {e}")
    
    return jsonify({
        "message": "saved",
        "file": annotation_filename,
        "next_file": next_file,
        "akshar": akshar_update,
        "duration": duration_update
    })


@app.route('/api/normal-skip-file', methods=['POST'])
def normal_skip_file():
    """Skip current normal file and get next one"""
    if not require_login():
        return jsonify({"error": "login required"}), 401

    username = session["user"]
    data = request.json
    current_file = data.get("current_file")
    category = get_user_normal_category_preference(username)

    # FIRST: Add current file to skip list and mark as pending
    if current_file:
        file_status = init_normal_file_status()
        
        # Track skips per user
        skip_key = f"normal_{username}"
        if skip_key not in user_skips:
            user_skips[skip_key] = []
        
        if current_file not in user_skips[skip_key]:
            user_skips[skip_key].append(current_file)
            print(f"Added {current_file} to skip list for {username}. Skip list: {user_skips[skip_key]}")
        
        # Release the file (set back to pending)
        if current_file in file_status:
            file_status[current_file]["status"] = "pending"
            file_status[current_file]["assigned_to"] = None
            file_status[current_file]["assigned_at"] = None
            save_normal_file_status(file_status)
            print(f"Released {current_file} back to pending")
        
        # Clear auto-save for skipped file
        autosave_path = get_normal_autosave_path(username, current_file)
        if os.path.exists(autosave_path):
            os.remove(autosave_path)
    
    # SECOND: Get next file (this will skip files in the skip list)
    next_file = get_normal_next_file_for_user(username, category)
    
    # Check if it's a BEEJ file
    is_beej = current_file.startswith('BEEJ_') if current_file else False
    
    message = "file skipped"
    
    # Special message for BEEJ files
    if is_beej and not next_file:
        file_status = init_normal_file_status()
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

@app.route('/api/debug-normal-skips')
def debug_normal_skips():
    """Debug endpoint to check normal file skips"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    skip_key = f"normal_{username}"
    skipped_files = user_skips.get(skip_key, [])
    
    file_status = init_normal_file_status()
    
    # Get all available files
    all_files = []
    for filename, status in file_status.items():
        if status.get("category") == get_user_normal_category_preference(username) or get_user_normal_category_preference(username) == 'all':
            all_files.append({
                "filename": filename,
                "status": status.get("status"),
                "assigned_to": status.get("assigned_to"),
                "priority": status.get("priority"),
                "category": status.get("category"),
                "is_skipped": filename in skipped_files
            })
    
    return jsonify({
        "username": username,
        "skip_list": skipped_files,
        "skip_count": len(skipped_files),
        "all_files": all_files,
        "current_category": get_user_normal_category_preference(username)
    })


@app.route('/api/normal-user-progress')
def get_normal_user_progress():
    """Get user progress for normal files"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    category = get_user_normal_category_preference(username)
    file_status = init_normal_file_status()
    user_completed = get_normal_user_completed_files(username)
    
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
        and f.get("assigned_to") == username
    )
    
    # Current file within category - check assigned files that are not completed
    current_file = None
    for filename, status in filtered_files.items():
        if (status.get("assigned_to") == username and 
            status.get("status") == "assigned" and
            filename not in user_completed):
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

# Add to navigation in index.html


# ==============================
# NORMAL FILE VERIFICATION ROUTES
# ==============================

@app.route('/api/normal-annotator-files/<username>')
def get_normal_annotator_files(username):
    """Get all completed normal files for a specific annotator"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    user_dir = get_normal_user_annotation_dir(username)
    if not os.path.exists(user_dir):
        return jsonify({"files": []})
    
    file_status = init_normal_file_status()
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

@app.route('/api/normal-load-for-verification/<username>/<filename>')
def normal_load_for_verification(username, filename):
    """Load a completed normal annotation for verification"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    # Load the completed annotation JSON from normal annotator's folder
    annotation_path = os.path.join(NORMAL_ANNOTATIONS_FOLDER, username, f"{os.path.splitext(filename)[0]}.json")
    
    if not os.path.exists(annotation_path):
        return jsonify({"error": "Annotation file not found"}), 404
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # Get audio file path
        audio_path = find_normal_audio_file(filename)
        
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
        print(f"Error loading normal file for verification: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/normal-verify-submit', methods=['POST'])
def normal_verify_submit():
    """Submit verification for a normal file"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    filename = data.get("filename")
    username = data.get("annotator")
    frames = data.get("frames", [])
    verified_by = session["user"]
    
    if not filename or not username:
        return jsonify({"error": "Missing data"}), 400
    
    # Create verified folder for normal files
    NORMAL_VERIFIED_FOLDER = "normal_verified"
    os.makedirs(NORMAL_VERIFIED_FOLDER, exist_ok=True)
    
    # Save verified annotation
    base = os.path.splitext(filename)[0]
    verified_filename = f"{base}.json"
    verified_path = os.path.join(NORMAL_VERIFIED_FOLDER, verified_filename)
    
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
    
    # Get audio file duration
    audio_path = find_normal_audio_file(filename)
    if audio_path and os.path.exists(audio_path):
        info = sf.info(audio_path)
        duration = info.duration
    else:
        duration = frames[-1]["end_ms"] / 1000.0 if frames else 0
    
    # Generate enhanced TextGrid for normal file
    tg_filename = f"{base}.TextGrid"
    tg_path = os.path.join(NORMAL_VERIFIED_FOLDER, tg_filename)
    
    tg_content = create_enhanced_normal_textgrid(
        frames, 
        duration, 
        data.get("full_sequence", ""), 
        username,
        window_ms=108
    )
    
    with open(tg_path, "w", encoding="utf-8") as f:
        f.write(tg_content)
    
    # Save to NORMAL_UI_DATASET
    ui_tg_path = os.path.join(NORMAL_UI_DATASET_DIR, f"{base}.TextGrid")
    with open(ui_tg_path, "w", encoding="utf-8") as f:
        f.write(tg_content)
    
    # Update normal_file_status.json to mark as verified
    file_status = init_normal_file_status()
    if filename in file_status:
        file_status[filename]["verified"] = True
        file_status[filename]["verified_by"] = verified_by
        file_status[filename]["verified_at"] = datetime.now().isoformat()
        file_status[filename]["verification_file"] = verified_filename
        file_status[filename]["verification_tg"] = tg_filename
        save_normal_file_status(file_status)
    
    # =========================
    # 🔥 NEW: DATE-WISE ORGANIZED SAVING (不影响现有功能)
    # =========================
    try:
        save_to_date_wise_verified(
            original_filename=filename,
            original_annotator=username,
            verified_by=verified_by,
            frames=frames,
            sentence=data.get("sentence", ""),
            full_sequence=data.get("full_sequence", ""),
            file_type='normal'
        )
    except Exception as e:
        print(f"Warning: Date-wise normal_verified save failed (不影响主流程): {e}")
    
    return jsonify({
        "success": True,
        "message": "Normal file verified successfully",
        "verified_file": verified_filename,
        "tg": tg_filename
    })

@app.route('/api/normal-verify-reject', methods=['POST'])
def normal_verify_reject():
    """Reject a completed normal annotation and mark it as pending for re-annotation"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    data = request.json
    filename = data.get("filename")
    annotator = data.get("annotator")
    rejected_by = data.get("rejected_by", session["user"])
    
    if not filename or not annotator:
        return jsonify({"error": "Missing data"}), 400
    
    file_status = init_normal_file_status()
    
    if filename not in file_status:
        return jsonify({"error": "File not found in status"}), 404
    
    # Load annotation to get stats
    annotation_file = file_status[filename].get("annotation_file")
    frames = []
    duration_seconds = 0
    
    if annotation_file:
        annotation_path = os.path.join(NORMAL_ANNOTATIONS_FOLDER, annotator, annotation_file)
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                    frames = annotation_data.get("frames", [])
                
                audio_path = find_normal_audio_file(filename)
                if audio_path and os.path.exists(audio_path):
                    info = sf.info(audio_path)
                    duration_seconds = info.duration
            except Exception as e:
                print(f"Error reading annotation file: {e}")
    
    akshar_count = sum(1 for frame in frames if frame.get("text") and frame["text"].strip() != "")
    
    # Reverse stats
    if akshar_count > 0 or duration_seconds > 0:
        reverse_user_stats(annotator, akshar_count, duration_seconds)
    
    # Move annotation files to rejected folder
    REJECTED_FOLDER = "rejected_normal_annotations"
    os.makedirs(REJECTED_FOLDER, exist_ok=True)
    
    deleted_files = []
    
    if annotation_file:
        annotation_path = os.path.join(NORMAL_ANNOTATIONS_FOLDER, annotator, annotation_file)
        if os.path.exists(annotation_path):
            rejected_path = os.path.join(REJECTED_FOLDER, f"{annotator}_{annotation_file}")
            os.rename(annotation_path, rejected_path)
            deleted_files.append(f"Moved: {annotation_file}")
        
        base = os.path.splitext(filename)[0]
        tg_path = os.path.join(NORMAL_ANNOTATIONS_FOLDER, annotator, f"{base}.TextGrid")
        if os.path.exists(tg_path):
            rejected_tg_path = os.path.join(REJECTED_FOLDER, f"{annotator}_{base}.TextGrid")
            os.rename(tg_path, rejected_tg_path)
            deleted_files.append(f"Moved: {base}.TextGrid")
    
    # Clean up verified files
    NORMAL_VERIFIED_FOLDER = "normal_verified"
    verified_json = os.path.join(NORMAL_VERIFIED_FOLDER, f"{os.path.splitext(filename)[0]}.json")
    verified_tg = os.path.join(NORMAL_VERIFIED_FOLDER, f"{os.path.splitext(filename)[0]}.TextGrid")
    
    for file_path in [verified_json, verified_tg]:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_files.append(f"Deleted: {os.path.basename(file_path)}")
    
    # Remove from NORMAL_UI_DATASET
    base = os.path.splitext(filename)[0]
    ui_tg_path = os.path.join(NORMAL_UI_DATASET_DIR, f"{base}.TextGrid")
    if os.path.exists(ui_tg_path):
        os.remove(ui_tg_path)
        deleted_files.append(f"Deleted from NORMAL_UI_DATASET: {base}.TextGrid")
    
    # Update file status
    file_status[filename]["status"] = "pending"
    file_status[filename]["assigned_to"] = None
    file_status[filename]["assigned_at"] = None
    file_status[filename]["completed_at"] = None
    file_status[filename]["annotation_file"] = None
    file_status[filename]["verified"] = False
    file_status[filename]["verified_by"] = None
    file_status[filename]["verified_at"] = None
    
    save_normal_file_status(file_status)
    
    # Clear autosave
    regular_autosave_path = get_normal_autosave_path(annotator, filename)
    if os.path.exists(regular_autosave_path):
        os.remove(regular_autosave_path)
    
    return jsonify({
        "success": True,
        "message": f"Normal file rejected. Removed {akshar_count} akshars and {duration_seconds:.1f}s from {annotator}'s stats",
        "akshar_removed": akshar_count,
        "duration_removed": duration_seconds,
        "deleted_files": deleted_files
    })



# ==============================
# NORMAL FILE VERIFICATION AUTO-SAVE ROUTES
# ==============================

def get_normal_verification_autosave_path(username, filename):
    """Get path for normal file verification auto-save"""
    user_autosave_dir = os.path.join(AUTOSAVE_FOLDER, "normal_verification", username)
    os.makedirs(user_autosave_dir, exist_ok=True)
    
    base = os.path.splitext(filename)[0]
    autosave_filename = f"{base}_verification_autosave.json"
    return os.path.join(user_autosave_dir, autosave_filename)

@app.route('/api/normal-verification-autosave', methods=['POST'])
def normal_verification_autosave():
    """Auto-save normal file verification progress"""
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
    
    autosave_path = get_normal_verification_autosave_path(username, audio_file)
    
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

@app.route('/api/normal-verification-autosave/<annotator>/<filename>', methods=['GET'])
def get_normal_verification_autosave(annotator, filename):
    """Get auto-saved normal file verification progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_normal_verification_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        with open(autosave_path, 'r', encoding='utf-8') as f:
            autosave_data = json.load(f)
        return jsonify(autosave_data)
    
    return jsonify({"frames": [], "edited_cells": []})

@app.route('/api/normal-verification-autosave/clear/<annotator>/<filename>', methods=['POST'])
def clear_normal_verification_autosave(annotator, filename):
    """Clear auto-saved normal file verification progress"""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    
    username = session["user"]
    autosave_path = get_normal_verification_autosave_path(username, filename)
    
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    
    return jsonify({"message": "autosave cleared"})


# ==============================
# SELF-RECORDED ANNOTATION ROUTES (with Auto-Save & Session Recovery)
# ==============================

# Constants
UI_RECORDING_DATA_FOLDER = "UI_RECORDING_DATA"
SELF_RECORD_AUTOSAVE_FOLDER = "self_record_autosave"
SELF_RECORD_SESSION_FOLDER = "self_record_sessions"

# Create folders if they don't exist
os.makedirs(UI_RECORDING_DATA_FOLDER, exist_ok=True)
os.makedirs(SELF_RECORD_AUTOSAVE_FOLDER, exist_ok=True)
os.makedirs(SELF_RECORD_SESSION_FOLDER, exist_ok=True)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def get_user_ui_recording_folder(username):
    """Get user-specific folder for UI recordings (original and slowed)."""
    user_folder = os.path.join(UI_RECORDING_DATA_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def get_self_record_autosave_path(username, filename):
    """Get path for auto‑save file (cell annotations and sentence)."""
    user_autosave_dir = os.path.join(SELF_RECORD_AUTOSAVE_FOLDER, username)
    os.makedirs(user_autosave_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]
    autosave_filename = f"{base}_autosave.json"
    return os.path.join(user_autosave_dir, autosave_filename)

def get_self_record_session_path(username):
    """Get path for session metadata (speed, frame size, slowed filename)."""
    return os.path.join(SELF_RECORD_SESSION_FOLDER, f"{username}_session.json")

def convert_webm_to_wav(webm_bytes, output_path, target_sr=16000):
    """Convert WebM audio bytes to WAV file using ffmpeg (fallback to scipy)."""
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            temp_webm.write(webm_bytes)
            temp_webm_path = temp_webm.name

        cmd = [
            'ffmpeg', '-i', temp_webm_path,
            '-ar', str(target_sr),
            '-ac', '1',
            '-c', 'pcm_s16le',
            '-y',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(temp_webm_path)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        return True
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        # Fallback: use soundfile and scipy (if available)
        try:
            import soundfile as sf
            import numpy as np
            from scipy import signal
            # Convert webm to wav using scipy? Not directly; fallback to reading as raw? Better to raise.
            # For simplicity, we re-raise because without ffmpeg we cannot decode webm.
            raise Exception("No ffmpeg available and webm conversion not implemented with scipy alone.")
        except:
            return False

# ------------------------------------------------------------------
# Page route
# ------------------------------------------------------------------
@app.route('/self-record-annotate')
def self_record_annotate_page():
    if not require_login():
        return redirect("/login")
    return render_template("self_record_annotate.html", user=session["user"])

@app.route('/ui-recording-audio/<username>/<filename>')
def serve_ui_recording_audio(username, filename):
    if not require_login():
        return redirect("/login")
    if session.get("user") != username:
        return jsonify({"error": "Unauthorized"}), 403
    user_folder = get_user_ui_recording_folder(username)
    filepath = os.path.join(user_folder, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='audio/wav')
    return jsonify({"error": "File not found"}), 404

# ------------------------------------------------------------------
# Prepare slowed audio (called when user clicks "Annotate Recording")
# ------------------------------------------------------------------
@app.route('/api/prepare-slowed-audio', methods=['POST'])
def prepare_slowed_audio():
    if not require_login():
        return jsonify({"success": False, "error": "not logged in"}), 401

    username = session["user"]
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    original_filename = request.form.get('filename')
    speed_factor = int(request.form.get('speed_factor', 2))
    speed = request.form.get('speed', '2x')
    frame_size = float(request.form.get('frame_size', 0.108))

    if not original_filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"{username}_{timestamp}.wav"

    user_folder = get_user_ui_recording_folder(username)
    original_path = os.path.join(user_folder, original_filename)

    # Convert uploaded WebM to WAV
    audio_bytes = audio_file.read()
    if not convert_webm_to_wav(audio_bytes, original_path, 16000):
        return jsonify({"success": False, "error": "Failed to convert audio to WAV"}), 500

    # Create slowed version (2x or 4x slower)
    slowed_filename = f"{os.path.splitext(original_filename)[0]}_{speed}.wav"
    slowed_path = os.path.join(user_folder, slowed_filename)

    try:
        import subprocess
        cmd = ['sox', original_path, slowed_path, 'tempo', str(1.0 / speed_factor)]
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        # Fallback to scipy/numpy
        try:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(original_path)
            if speed_factor == 2:
                slowed_data = np.repeat(data, 2, axis=0)
            elif speed_factor == 4:
                slowed_data = np.repeat(data, 4, axis=0)
            else:
                slowed_data = data
            sf.write(slowed_path, slowed_data, sr)
        except Exception as e2:
            return jsonify({"success": False, "error": f"Failed to slow audio: {e2}"}), 500

    # Save session metadata for recovery after page refresh
    session_path = get_self_record_session_path(username)
    session_data = {
        "filename": slowed_filename,
        "speed": speed,
        "speed_factor": speed_factor,
        "frame_size": frame_size,
        "timestamp": datetime.now().isoformat()
    }
    with open(session_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2)

    return jsonify({
        "success": True,
        "slowed_filename": slowed_filename,
        "slowed_audio_url": f"/ui-recording-audio/{username}/{slowed_filename}",
        "original_filename": original_filename,
        "speed_factor": speed_factor,
        "speed": speed,
        "frame_size": frame_size
    })

# ------------------------------------------------------------------
# Submit final annotation (both slowed and normal versions)
# ------------------------------------------------------------------
@app.route('/api/submit-self-recorded', methods=['POST'])
def submit_self_recorded():
    if not require_login():
        return jsonify({"success": False, "error": "not logged in"}), 401

    username = session["user"]
    slowed_filename = request.form.get('slowed_filename')
    speed_factor = int(request.form.get('speed_factor', 2))
    speed = request.form.get('speed', '2x')
    frame_size = float(request.form.get('frame_size', 0.108))
    frames_json = request.form.get('frames', '[]')
    sentence_text = request.form.get('sentence', '')

    if not slowed_filename:
        return jsonify({"success": False, "error": "Missing slowed_filename"}), 400

    try:
        frames = json.loads(frames_json)
    except:
        frames = []

    user_folder = get_user_ui_recording_folder(username)
    slowed_path = os.path.join(user_folder, slowed_filename)

    # 🔥 The slowed audio should already exist from prepare-slowed-audio
    if not os.path.exists(slowed_path):
        return jsonify({"success": False, "error": "Slowed audio file not found. Please re-record."}), 404

    # No need to save the file again – it's already there.
    print(f"Using existing slowed audio: {slowed_path}")

    base = slowed_filename.replace(f"_{speed}", "").replace(".wav", "")
    original_filename = f"{base}.wav"
    original_path = os.path.join(user_folder, original_filename)

    akshar_count = sum(1 for f in frames if f.get("text") and f["text"].strip())

    # ---------- Slowed version JSON ----------
    slowed_json_filename = f"{base}_{speed}.json"
    slowed_json_path = os.path.join(user_folder, slowed_json_filename)
    try:
        import soundfile as sf
        slowed_duration = sf.info(slowed_path).duration
    except:
        slowed_duration = len(frames) * frame_size

    slowed_annotation = {
        "audio_file": slowed_filename,
        "annotator": username,
        "timestamp": datetime.now().isoformat(),
        "window_ms": int(frame_size * 1000),
        "sentence": sentence_text,
        "full_sequence": sentence_text.replace(" ", ""),
        "frames": frames,
        "category": f"self_recorded_{speed}",
        "self_recorded": True,
        "speed": speed,
        "speed_factor": speed_factor,
        "akshar_count": akshar_count
    }
    with open(slowed_json_path, 'w', encoding='utf-8') as f:
        json.dump(slowed_annotation, f, indent=2)

    # ---------- Slowed TextGrid ----------
    def create_textgrid(frames, duration, sentence, annotator, window_ms):
        tg = []
        tg.append('File type = "ooTextFile"')
        tg.append('Object class = "TextGrid"\n')
        tg.append(f"xmin = 0")
        tg.append(f"xmax = {duration}")
        tg.append("tiers? <exists>")
        tg.append("size = 3")
        tg.append("item []:")
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
        tg.append("    item [2]:")
        tg.append('        class = "IntervalTier"')
        tg.append('        name = "annotations"')
        tg.append(f"        xmin = 0")
        tg.append(f"        xmax = {duration}")
        tg.append(f"        intervals: size = {len(frames)}")
        for i, f in enumerate(frames, 1):
            start = f.get("start_ms", i * window_ms) / 1000.0
            end = f.get("end_ms", (i + 1) * window_ms) / 1000.0
            text = f.get("text", "") or ""
            tg.append(f"        intervals [{i}]:")
            tg.append(f"            xmin = {start}")
            tg.append(f"            xmax = {end}")
            tg.append(f'            text = "{text}"')
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

    slowed_tg_path = os.path.join(user_folder, f"{base}_{speed}.TextGrid")
    with open(slowed_tg_path, 'w', encoding='utf-8') as f:
        f.write(create_textgrid(frames, slowed_duration, sentence_text, username, int(frame_size * 1000)))

    # ---------- Normal version (scale frames) ----------
    normal_frame_size = frame_size / speed_factor
    normal_frames = []
    for f in frames:
        normal_frames.append({
            "index": f["index"],
            "start_ms": f["start_ms"] // speed_factor,
            "end_ms": f["end_ms"] // speed_factor,
            "text": f["text"]
        })

    try:
        import soundfile as sf
        if os.path.exists(original_path):
            normal_duration = sf.info(original_path).duration
        else:
            normal_duration = slowed_duration / speed_factor
    except:
        normal_duration = slowed_duration / speed_factor

    # Normal JSON
    normal_json_path = os.path.join(user_folder, f"{base}.json")
    normal_annotation = {
        "audio_file": original_filename,
        "annotator": username,
        "timestamp": datetime.now().isoformat(),
        "window_ms": int(normal_frame_size * 1000),
        "sentence": sentence_text,
        "full_sequence": sentence_text.replace(" ", ""),
        "frames": normal_frames,
        "category": "self_recorded_normal",
        "self_recorded": True,
        "speed": "normal",
        "original_speed": speed,
        "akshar_count": akshar_count
    }
    with open(normal_json_path, 'w', encoding='utf-8') as f:
        json.dump(normal_annotation, f, indent=2)

    # Normal TextGrid
    normal_tg_path = os.path.join(user_folder, f"{base}.TextGrid")
    with open(normal_tg_path, 'w', encoding='utf-8') as f:
        f.write(create_textgrid(normal_frames, normal_duration, sentence_text, username, int(normal_frame_size * 1000)))

    # ========== EXTRA SAVE TO UI_RECORDING_NORMAL_DATA ==========
    import shutil
    UI_RECORDING_NORMAL_DATA = "UI_RECORDING_NORMAL_DATA"
    os.makedirs(UI_RECORDING_NORMAL_DATA, exist_ok=True)
    
    if os.path.exists(original_path):
        shutil.copy2(original_path, os.path.join(UI_RECORDING_NORMAL_DATA, original_filename))
        print(f"Extra copy of normal WAV saved to UI_RECORDING_NORMAL_DATA/{original_filename}")
    
    if os.path.exists(normal_tg_path):
        shutil.copy2(normal_tg_path, os.path.join(UI_RECORDING_NORMAL_DATA, f"{base}.TextGrid"))
        print(f"Extra copy of normal TextGrid saved to UI_RECORDING_NORMAL_DATA/{base}.TextGrid")
    
    if os.path.exists(normal_json_path):
        shutil.copy2(normal_json_path, os.path.join(UI_RECORDING_NORMAL_DATA, f"{base}.json"))
        print(f"Extra copy of normal JSON saved to UI_RECORDING_NORMAL_DATA/{base}.json")
    # ============================================================

    # Update stats
    akshar_update = update_akshar_counts(username, frames)
    duration_update = update_duration_counts(username, normal_duration)

    # Cleanup auto-save and session
    autosave_path = get_self_record_autosave_path(username, slowed_filename)
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    session_path = get_self_record_session_path(username)
    if os.path.exists(session_path):
        os.remove(session_path)

    return jsonify({
        "success": True,
        "slowed_filename": slowed_filename,
        "normal_filename": original_filename,
        "akshar_added": akshar_count,
        "akshar": akshar_update,
        "duration": duration_update,
        "message": "Self-recorded annotation saved successfully"
    })



# ------------------------------------------------------------------
# List user's self-recorded normal files (optional)
# ------------------------------------------------------------------
@app.route('/api/user-self-recordings')
def get_user_self_recordings():
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    username = session["user"]
    user_folder = get_user_ui_recording_folder(username)
    recordings = []
    if os.path.exists(user_folder):
        for fname in os.listdir(user_folder):
            if fname.endswith('.wav') and not ('_2x' in fname or '_4x' in fname):
                filepath = os.path.join(user_folder, fname)
                json_path = filepath.replace('.wav', '.json')
                try:
                    stat = os.stat(filepath)
                    import soundfile as sf
                    info = sf.info(filepath)
                    akshar_count = 0
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                            akshar_count = sum(1 for f in data.get("frames", []) if f.get("text"))
                    recordings.append({
                        "filename": fname,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "duration": round(info.duration, 2),
                        "akshar_count": akshar_count,
                        "has_annotation": os.path.exists(json_path)
                    })
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
        recordings.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify({"success": True, "recordings": recordings})

# ------------------------------------------------------------------
# Auto-save API (cell changes)
# ------------------------------------------------------------------
@app.route('/api/self-record-autosave', methods=['POST'])
def self_record_autosave():
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    data = request.json
    username = session["user"]
    filename = data.get("filename")
    frames = data.get("frames", [])
    speed = data.get("speed", "2x")
    speed_factor = data.get("speed_factor", 2)
    frame_size = data.get("frame_size", 0.108)
    sentence = data.get("sentence", "")
    if not filename:
        return jsonify({"error": "no filename"}), 400
    autosave_path = get_self_record_autosave_path(username, filename)
    autosave_data = {
        "filename": filename,
        "annotator": username,
        "last_updated": datetime.now().isoformat(),
        "frames": frames,
        "speed": speed,
        "speed_factor": speed_factor,
        "frame_size": frame_size,
        "sentence": sentence
    }
    with open(autosave_path, 'w', encoding='utf-8') as f:
        json.dump(autosave_data, f, indent=2)
    return jsonify({"message": "autosaved"})

@app.route('/api/self-record-autosave/<filename>', methods=['GET'])
def get_self_record_autosave(filename):
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    username = session["user"]
    autosave_path = get_self_record_autosave_path(username, filename)
    if os.path.exists(autosave_path):
        with open(autosave_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"frames": [], "sentence": ""})

@app.route('/api/self-record-autosave/clear/<filename>', methods=['POST'])
def clear_self_record_autosave(filename):
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    username = session["user"]
    autosave_path = get_self_record_autosave_path(username, filename)
    if os.path.exists(autosave_path):
        os.remove(autosave_path)
    return jsonify({"message": "cleared"})

# ------------------------------------------------------------------
# Session recovery (used after page refresh)
# ------------------------------------------------------------------
@app.route('/api/self-record-session', methods=['GET'])
def get_self_record_session():
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    username = session["user"]
    session_path = get_self_record_session_path(username)
    if os.path.exists(session_path):
        with open(session_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({
            "has_session": True,
            "filename": data.get("filename"),
            "timestamp": data.get("timestamp")
        })
    return jsonify({"has_session": False})

@app.route('/api/self-record-session/<filename>', methods=['GET'])
def get_self_record_session_data(filename):
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    username = session["user"]
    user_folder = get_user_ui_recording_folder(username)
    slowed_path = os.path.join(user_folder, filename)
    if not os.path.exists(slowed_path):
        return jsonify({"success": False, "error": "Audio file not found"}), 404
    session_path = get_self_record_session_path(username)
    session_data = {}
    if os.path.exists(session_path):
        with open(session_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
    return jsonify({
        "success": True,
        "filename": filename,
        "slowed_audio_url": f"/ui-recording-audio/{username}/{filename}",
        "speed": session_data.get("speed", "2x"),
        "speed_factor": session_data.get("speed_factor", 2),
        "frame_size": session_data.get("frame_size", 0.108)
    })

@app.route('/api/self-record-session/clear', methods=['POST'])
def clear_self_record_session():
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    username = session["user"]
    session_path = get_self_record_session_path(username)
    if os.path.exists(session_path):
        os.remove(session_path)
    return jsonify({"success": True})


# ==============================
# DOWNLOAD_N_STORE ROUTES
# ==============================

DOWNLOAD_N_STORE_FOLDER = "DOWNLOAD_N_STORE"
os.makedirs(DOWNLOAD_N_STORE_FOLDER, exist_ok=True)

@app.route('/api/stored-files')
def get_stored_files():
    """List files in DOWNLOAD_N_STORE folder with metadata."""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    files = []
    try:
        for filename in os.listdir(DOWNLOAD_N_STORE_FOLDER):
            filepath = os.path.join(DOWNLOAD_N_STORE_FOLDER, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                size_bytes = stat.st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024*1024:
                    size_str = f"{size_bytes/1024:.1f} KB"
                else:
                    size_str = f"{size_bytes/(1024*1024):.1f} MB"
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                files.append({
                    "name": filename,
                    "size": size_bytes,
                    "size_formatted": size_str,
                    "modified": stat.st_mtime,
                    "modified_formatted": modified
                })
        files.sort(key=lambda x: x["modified"], reverse=True)
        return jsonify({"success": True, "files": files})
    except Exception as e:
        print(f"Error listing stored files: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download-stored-file/<filename>')
def download_stored_file(filename):
    """Download a single file from DOWNLOAD_N_STORE."""
    if not require_login():
        return redirect("/login")
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DOWNLOAD_N_STORE_FOLDER, safe_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=safe_name)
    return jsonify({"error": "File not found"}), 404

@app.route('/api/download-stored-batch', methods=['POST'])
def download_stored_batch():
    """Download multiple files as ZIP using a temporary file (handles large files)."""
    if not require_login():
        return jsonify({"error": "not logged in"}), 401
    data = request.json
    filenames = data.get("files", [])
    if not filenames:
        return jsonify({"error": "No files selected"}), 400

    import tempfile
    import zipfile
    import shutil

    # Create a temporary file for the ZIP
    temp_fd, temp_path = tempfile.mkstemp(suffix='.zip')
    os.close(temp_fd)

    try:
        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in filenames:
                safe_name = os.path.basename(fname)
                file_path = os.path.join(DOWNLOAD_N_STORE_FOLDER, safe_name)
                if os.path.exists(file_path):
                    zf.write(file_path, safe_name)
        # Send the ZIP file
        return send_file(temp_path, as_attachment=True, download_name='stored_files.zip', mimetype='application/zip')
    except Exception as e:
        print(f"Batch zip error: {e}")
        return jsonify({"error": "Failed to create zip"}), 500
    finally:
        # Clean up temporary file after sending (Flask will have read it)
        try:
            os.unlink(temp_path)
        except:
            pass


# Initialize file status on startup
init_file_status()
load_akshar_tracking()
load_duration_tracking()

init_normal_file_status()

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