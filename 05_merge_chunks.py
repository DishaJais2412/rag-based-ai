"""
Sliding Window Chunk Merging Pipeline.

Merges small, fine-grained Whisper segments into coherent, 45-second sliding-window
chunks with a 10-second overlap to eliminate abrupt sentence breaks and maintain
instructional context. Saves processed chunks to 'newjsons/'.
"""

import json
import os

TARGET_DURATION = 45   # Target window duration in seconds
OVERLAP_DURATION = 10  # Overlap between consecutive windows in seconds

os.makedirs("newjsons", exist_ok=True)

for filename in os.listdir("jsons"):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join("jsons", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        continue

    new_chunks = []
    start_index = 0

    # Dynamic sliding window across transcript segments
    while start_index < len(chunks):
        start_time = chunks[start_index]["start"]
        end_index = start_index

        # Expand segment window until target duration is reached
        while end_index < len(chunks):
            current_end = chunks[end_index]["end"]
            if current_end - start_time >= TARGET_DURATION:
                break
            end_index += 1

        # Prevent out-of-bounds indexing
        if end_index >= len(chunks):
            end_index = len(chunks) - 1

        chunk_group = chunks[start_index:end_index + 1]

        # Combine text and boundaries for this consolidated chunk
        new_chunks.append({
            "number": chunk_group[0]["number"],
            "title": chunk_group[0]["title"],
            "start": chunk_group[0]["start"],
            "end": chunk_group[-1]["end"],
            "text": " ".join(c["text"].strip() for c in chunk_group)
        })

        # Calculate overlap for the next chunk window
        next_start_time = chunk_group[-1]["end"] - OVERLAP_DURATION
        next_start_index = end_index + 1

        while (
            next_start_index > start_index
            and next_start_index < len(chunks)
            and chunks[next_start_index]["start"] > next_start_time
        ):
            next_start_index -= 1

        # Fallback guard to guarantee forward progression
        if next_start_index <= start_index:
            next_start_index = end_index + 1

        start_index = next_start_index

    # Save merged chunks with original full transcript
    output_path = os.path.join("newjsons", filename)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(
            {
                "chunks": new_chunks,
                "text": data.get("text", "")
            },
            json_file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"{filename}: {len(chunks)} original segments -> {len(new_chunks)} merged chunks"
    )
