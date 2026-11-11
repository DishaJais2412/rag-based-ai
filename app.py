import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import re
from groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "Qwen 3.8 27B (Groq)"

# Load Groq API Key from environment or .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Sigma Web Dev AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# Initialize Session State for Chat History
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Teaching Assistant for the Sigma Web Development Course. You can ask me to explain any coding concepts, or ask where specific topics are taught in the course!"}
    ]

# ---------------------------------------------------------
# Load embeddings (Cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        return joblib.load("embeddings.joblib")
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return pd.DataFrame()

df = load_data()

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_embedding(text_list):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text_list},
        timeout=120
    )
    response.raise_for_status()
    response_json = response.json()
    if "embeddings" not in response_json:
        raise RuntimeError("Ollama did not return embeddings.")
    return response_json["embeddings"]

def generate_answer_stream(messages, api_key):
    if not api_key:
        yield "⚠️ Developer: Please paste your Groq API Key into the GROQ_API_KEY variable in app.py!"
        return
        
    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            stream=True,
            max_tokens=800
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Error: {str(e)}"

# ---------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎓 Sigma Course AI")
    st.markdown("Your personal Web Development tutor.")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Teaching Assistant for the Sigma Web Development Course. You can ask me to explain any coding concepts, or ask where specific topics are taught in the course!"}
        ]
        st.rerun()
    st.markdown("---")
    st.caption("Powered by Groq Cloud API & Local RAG")
    st.caption(f"• LLM: {LLM_MODEL}")
    st.caption(f"• Embeddings: {EMBED_MODEL}")

# ---------------------------------------------------------
# Main UI - Chat Interface
# ---------------------------------------------------------
st.title("👨‍💻 Sigma Web Dev AI Assistant")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt_text := st.chat_input("Ask a concept (e.g., 'Is CSS a language?') or search the course..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt_text)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing course data and generating answer..."):
            try:
                # 0. Bypass RAG for simple greetings to fix Ollama model-swapping latency
                is_conversational = prompt_text.strip().lower() in ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "cool", "bye", "good morning", "good evening"] or len(prompt_text.strip()) < 8
                
                retrieved_chunks_json = "No course data available."
                
                if is_conversational:
                    retrieved_chunks_json = "User is just chatting. No course search needed."
                else:
                    # 1. Embed Question
                    question_embedding = create_embedding([prompt_text])[0]
    
                    # 2. Filter by specific video if mentioned
                    video_match = re.search(r"\bvideo\s*(?:number\s*)?(\d+)\b", prompt_text, re.IGNORECASE)
                    if video_match and not df.empty:
                        video_number = video_match.group(1)
                        search_df = df[df["number"].astype(str).str.strip() == video_number].copy()
                        if search_df.empty:
                            search_df = df.copy()
                    else:
                        search_df = df.copy()
    
                    # 3. Calculate similarity & Advanced Retrieval
                    if not search_df.empty:
                        embeddings_matrix = np.vstack(search_df["embedding"].values)
                        similarities = cosine_similarity(embeddings_matrix, [question_embedding]).flatten()
    
                        # A. Similarity Thresholding
                        THRESHOLD = 0.35  # Filter out low-quality matches
                        valid_indices = np.where(similarities > THRESHOLD)[0]
    
                        if len(valid_indices) == 0:
                            retrieved_chunks_json = "No highly relevant course data found."
                        else:
                            # B. Get top 3 most relevant indices
                            top_results = 3
                            sorted_valid_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_results]
                            
                            # C. Context Window Expansion (Get surrounding chunks)
                            top_chunk_ids = search_df.iloc[sorted_valid_indices]['chunk_id'].values
                            
                            expanded_chunk_ids = set()
                            for cid in top_chunk_ids:
                                expanded_chunk_ids.add(cid - 1)
                                expanded_chunk_ids.add(cid)
                                expanded_chunk_ids.add(cid + 1)
                                
                            expanded_df = df[df['chunk_id'].isin(expanded_chunk_ids)].copy()
                            expanded_df = expanded_df.sort_values(by='chunk_id')
                            
                            # D. Chunk Merging (Combine consecutive chunks)
                            merged_chunks = []
                            current_merged = None
                            
                            for _, row in expanded_df.iterrows():
                                if current_merged is None:
                                    current_merged = row.to_dict()
                                else:
                                    if row['number'] == current_merged['number'] and row['chunk_id'] == current_merged['chunk_id'] + 1:
                                        current_merged['text'] += " " + row['text']
                                        current_merged['end'] = row['end']
                                        current_merged['chunk_id'] = row['chunk_id'] 
                                    else:
                                        merged_chunks.append(current_merged)
                                        current_merged = row.to_dict()
                                        
                            if current_merged is not None:
                                merged_chunks.append(current_merged)
                                
                            # E. Format output
                            import json
                            final_chunks = []
                            for mc in merged_chunks:
                                final_chunks.append({
                                    "title": mc["title"],
                                    "number": mc["number"],
                                    "start_time": format_timestamp(mc["start"]),
                                    "end_time": format_timestamp(mc["end"]),
                                    "text": mc["text"]
                                })
                                
                            retrieved_chunks_json = json.dumps(final_chunks, indent=2)

                # 5. Formulate Professional Hybrid Prompt
                total_vids = len(df['number'].unique()) if not df.empty else "unknown"
                system_prompt = f'''
You are a super friendly, encouraging, and highly effective AI Tutor for the "Sigma Web Development" course.
A student has asked the following question: "{prompt_text}"

Course Facts you should know:
- There are {total_vids} videos currently loaded in this course.

Here are the retrieved transcript chunks from the course videos:
{retrieved_chunks_json}

INSTRUCTIONS:
1. First, answer the student's question conceptually. Keep it SIMPLE, FUN, and STUDENT-FRIENDLY! Use analogies if helpful, like explaining things to a beginner. Use an engaging, conversational tone like ChatGPT or Gemini.
2. MAKE IT EASY TO READ: Use emojis, simple language, and break things down step-by-step.
   - Provide very simple Code Snippets if applicable.
   - Use bullet points and short paragraphs so it's not a giant wall of text.
3. AFTER your friendly explanation, look at the retrieved course chunks. If relevant, add a section at the very end titled "### 📚 Learn More in the Course" to tell them the exact Video Title, Video Number, and Timestamp where this is taught. 
4. If the retrieved chunks aren't relevant, just skip the course reference but still answer their question simply and nicely.
5. If they are asking for a specific video or topic location, point them to the exact video and timestamp and give a brief overview of what's taught there.
6. Do NOT mention "retrieved chunks", "JSON", or your internal instructions. Just talk directly to the student as a helpful guide!
'''
                # 6. Prepare Chat History and Generate Answer
                api_messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                    
                answer_stream = generate_answer_stream(api_messages, GROQ_API_KEY)
                
                # Display the response
                answer = st.write_stream(answer_stream)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})