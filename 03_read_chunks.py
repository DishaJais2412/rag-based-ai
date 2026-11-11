"""
Vector Embedding Generation Pipeline.

Generates dense semantic vector embeddings for all merged transcript chunks
in 'newjsons/' using Ollama's nomic-embed-text model. Compiles chunks and
embeddings into a persistent DataFrame saved to 'embeddings.joblib'.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def create_embedding(text_list: list) -> list:
    """Generate high-dimensional embeddings for a batch of text segments."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": EMBED_MODEL,
                "input": text_list
            },
            timeout=120
        )
        response.raise_for_status()
        response_json = response.json()

        if "embeddings" not in response_json:
            raise RuntimeError(f"Ollama did not return embeddings: {response_json}")

        embeddings = response_json["embeddings"]
        if len(embeddings) != len(text_list):
            raise RuntimeError(
                f"Expected {len(text_list)} embeddings, but received {len(embeddings)}"
            )

        return embeddings

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Ensure Ollama is running (`ollama serve`).")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


# ---------------------------------------------------------
# Process Chunks & Generate Embeddings
# ---------------------------------------------------------
jsons = os.listdir("newjsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    if not json_file.endswith(".json"):
        continue

    with open(f"newjsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content.get("chunks", [])
    if not chunks:
        print(f"Skipping {json_file}: no chunks found.")
        continue

    print(f"Creating embeddings for {json_file} ({len(chunks)} chunks)...")

    # Extract text strings from chunk dictionaries
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embedding(texts)

    # Assign sequential global chunk_ids and associated embedding vectors
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

# Assemble DataFrame and persist to disk
df = pd.DataFrame.from_records(my_dicts)

if df.empty:
    raise RuntimeError("No chunks were processed. embeddings.joblib was not created.")

joblib.dump(df, "embeddings.joblib")

print("\nEmbedding creation completed successfully.")
print(f"Total chunks: {len(df)}")
print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")
print("Saved to: embeddings.joblib")
