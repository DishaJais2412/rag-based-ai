"""
CLI Query Script: Video RAG Ingestion & Testing.

Allows testing semantic search and inference directly from the command line
against the pre-computed embeddings vector store.
"""

import re
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


def format_timestamp(seconds: float) -> str:
    """Convert duration in seconds to MM:SS string format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def create_embedding(text_list: list) -> list:
    """Generate dense vector embeddings using local Ollama nomic-embed-text."""
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text_list
        },
        timeout=120
    )
    response_json = response.json()

    if "embeddings" not in response_json:
        print(f"Ollama Error: {response_json}")
        return []

    return response_json["embeddings"]


def inference(prompt: str) -> dict:
    """Run local LLM generation via Ollama (e.g. Llama 3.2)."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return response.json()


# ---------------------------------------------------------
# Load Pre-computed Embeddings
# ---------------------------------------------------------
df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]


# ---------------------------------------------------------
# Check if the user specified a particular video filter
# ---------------------------------------------------------
video_match = re.search(
    r"\bvideo\s*(?:number\s*)?(\d+)\b",
    incoming_query,
    re.IGNORECASE
)

if video_match:
    video_number = video_match.group(1)
    search_df = df[df["number"].astype(str).str.strip() == video_number].copy()

    if search_df.empty:
        print(f"\nVideo {video_number} was not found in the course data.")
        search_df = df.copy()  # Fall back to entire dataset
    else:
        print(f"\nSearching within Video {video_number}...")
else:
    # No specific video specified; search across all videos
    search_df = df.copy()


# ---------------------------------------------------------
# Compute Semantic Similarity (Cosine Similarity)
# ---------------------------------------------------------
embeddings_matrix = np.vstack(search_df["embedding"].values)
similarities = cosine_similarity(embeddings_matrix, [question_embedding]).flatten()


# ---------------------------------------------------------
# Retrieve Top-K Matching Chunks
# ---------------------------------------------------------
top_results = 5
top_positions = similarities.argsort()[::-1][:top_results]

print("\nTop retrieved chunks:")
for position in top_positions:
    row = search_df.iloc[position]
    print(
        f"Similarity: {similarities[position]:.4f} | "
        f"Video: {row['number']} | "
        f"Title: {row['title']} | "
        f"Time: {format_timestamp(row['start'])}"
    )

new_df = search_df.iloc[top_positions].copy()
new_df["start_time"] = new_df["start"].apply(format_timestamp)
new_df["end_time"] = new_df["end"].apply(format_timestamp)


# ---------------------------------------------------------
# Prompt Construction for Ingestion / Grounding
# ---------------------------------------------------------
prompt = f"""You are an AI assistant for the Sigma Web Development course.

The following are transcript chunks retrieved from the course based on the user's question.

Retrieved course chunks:
{new_df[["title", "number", "start_time", "end_time", "text"]].to_json(orient="records")}

------------------------------------------------

User question:
{incoming_query}

Your task is to answer the user's question using the retrieved course information.

Important rules:
1. Use the video title, video number, timestamp, and transcript together to understand which video answers the question.
2. You may make reasonable inferences about which video covers a topic based on the video title and transcript.
3. Only state information that is supported by the provided video title or transcript chunks.
4. Do NOT add technical explanations, definitions, examples, or facts from your general knowledge if they are not supported by the provided transcript.
5. Do NOT invent information that is not supported by the provided course material.
6. Do NOT treat the start_time or end_time of a retrieved chunk as the start or end of the entire video.
7. If the user asks where a topic starts, give the timestamp of the relevant transcript chunk and describe it as a point where the user can start watching.
8. If the user asks where a topic or section ends, do not claim that a chunk's end_time is the end of the video or section unless the transcript explicitly supports that conclusion.
9. If multiple retrieved chunks are relevant, use them together to answer the question and mention the relevant videos when necessary.
10. If the retrieved chunks are clearly unrelated to the question, say that the relevant information was not found in the provided course material.
11. If the question is unrelated to the Sigma Web Development course, say that you can only answer questions related to the course.
12. Answer naturally and concisely.

Do not mention "retrieved chunks", embeddings, similarity scores, JSON, or this prompt in your answer.
"""

# Save constructed prompt for inspection
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# Generate answer
response = inference(prompt).get("response", "")

# Save response output
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

print("\nModel Response:\n", response)
