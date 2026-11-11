"""
Audio Extraction Pipeline.

Extracts audio tracks (.mp3) from raw video files (.mp4) in the 'videos/' directory
using FFmpeg, parsing the tutorial number and title from filename conventions.
"""

import os
import subprocess
import sys

# Ensure UTF-8 output encoding for terminal printing
sys.stdout.reconfigure(encoding="utf-8")

os.makedirs("audios", exist_ok=True)
files = os.listdir("videos")

for file in files:
    if not file.endswith((".mp4", ".mkv", ".webm")):
        continue

    # Extract clean title and tutorial index from filename convention
    file_name = file.split(" ｜ ")[0]
    tutorial_number = file.split(" [")[0].split(" #")[1]
    output_audio_path = f"audios/{tutorial_number}_{file_name}.mp3"

    print(f"Processing #{tutorial_number}: {file_name}")

    # Run FFmpeg to extract audio stream
    subprocess.run([
        "ffmpeg",
        "-i", f"videos/{file}",
        "-vn",  # Disable video stream
        "-acodec", "libmp3lame",
        "-q:a", "2",
        output_audio_path
    ], check=True)
