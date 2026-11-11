"""
Speech Transcription & Translation Pipeline.

Iterates over audio recordings in 'audios/', transcribes Hindi speech,
and translates it to English using OpenAI's Whisper model.
Saves raw segmented transcriptions to 'jsons/'.
"""

import json
import os
import whisper

# Load Whisper model (downloads to specified directory if not present)
model = whisper.load_model("medium", download_root="D:/whisper_models")

os.makedirs("jsons", exist_ok=True)
audios = os.listdir("audios")

for audio in audios:
    if "_" not in audio or not audio.endswith(".mp3"):
        continue

    # Extract tutorial metadata from filename
    number = audio.split("_")[0]
    title = audio.split("_")[1][:-4]
    print(f"Transcribing #{number}: {title}...")

    # Transcribe Hindi speech and translate to English text
    result = model.transcribe(
        audio=f"audios/{audio}",
        language="hi",
        task="translate",
        word_timestamps=False
    )

    # Collect individual speech segments with timing metadata
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    # Save to JSON file
    output_path = f"jsons/{audio}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} segments to {output_path}")
