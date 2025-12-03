import streamlit as st
from transformers import AlbertTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import tempfile
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --- CONFIG ---
MODEL_PATH = "model"  # Your model folder
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# --- LOAD MODEL + TOKENIZER ---
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# --- PREDICT FUNCTION ---
def predict_abuse(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    logits = model(**inputs).logits
    pred = np.argmax(logits, axis=1)[0]
    return "Abusive" if pred == 1 else "Safe"

# --- AUTHENTICATION FUNCTION ---
def authenticate_youtube(client_secret_path):
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
    credentials = flow.run_local_server(port=0)
    return build("youtube", "v3", credentials=credentials)

# --- FETCH COMMENTS FUNCTION ---
def fetch_comments(youtube, video_id, max_results=20):
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results)
    response = request.execute()
    comments = []
    for item in response.get("items", []):
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        cid = item["id"]
        comments.append((cid, text))
    return comments

# --- DELETE COMMENT FUNCTION ---
def delete_comment(youtube, comment_id):
    youtube.comments().setModerationStatus(id=comment_id, moderationStatus="rejected").execute()

# --- STREAMLIT UI ---
st.title("🧹 YouTube Abusive Comment Auto-Moderator")

# Upload client_secret.json
client_file = st.file_uploader("Upload your Google client_secret.json", type=["json"])
video_id = st.text_input("Enter YouTube Video ID")

if st.button("Fetch, Classify & Auto-Delete Abusive Comments"):
    if client_file is None or video_id.strip() == "":
        st.warning("Please upload client_secret.json and enter a Video ID.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(client_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Authenticating with YouTube..."):
                youtube = authenticate_youtube(tmp_path)

            with st.spinner("Fetching comments..."):
                comments = fetch_comments(youtube, video_id)
                st.success(f"✅ Fetched {len(comments)} comments.")

            st.subheader("🧪 Moderation Results")

            for cid, text in comments:
                label = predict_abuse(text)
                if label == "Abusive":
                    st.error(f"[Abusive] {text}")
                    try:
                        delete_comment(youtube, cid)
                        st.warning(f"🗑️ Deleted abusive comment: {cid}")
                    except Exception as e:
                        st.error(f"❌ Failed to delete comment {cid}: {e}")
                else:
                    st.info(f"[Safe] {text}")

        finally:
            os.remove(tmp_path)
