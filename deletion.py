import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable ONEDNN optimizations for consistent predictions

import tensorflow as tf
from transformers import AlbertTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ========== 1️⃣ Load Model and Tokenizer ==========
MODEL_PATH = "model"  # Folder containing tf_model.h5 and tokenizer files

print("Loading model and tokenizer...")
tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print("Model and tokenizer loaded ✅")

# ========== 2️⃣ Classify Comment ==========
def preprocess_and_predict(text):
    """Returns True if the comment is classified as abusive."""
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    logits = model(**inputs).logits
    prediction = np.argmax(logits, axis=1)[0]  # 0 = safe, 1 = abusive
    return prediction == 1

# ========== 3️⃣ YouTube Authentication ==========
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CLIENT_SECRET_FILE = "client_secret.json"

def authenticate_youtube():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    return build("youtube", "v3", credentials=credentials)

# ========== 4️⃣ Fetch Comments ==========
def fetch_comments(youtube, video_id, max_results=100):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    comments = []
    for item in response.get("items", []):
        comment_id = item["id"]
        comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append((comment_id, comment_text))
    return comments

# ========== 5️⃣ Delete Comment ==========
def delete_comment(youtube, comment_id):
    request = youtube.comments().setModerationStatus(
        id=comment_id,
        moderationStatus="rejected",
        banAuthor=False  # Set to True to ban the user as well
    )
    request.execute()
    print(f"❌ Deleted abusive comment: {comment_id}")

# ========== 6️⃣ Combine: Fetch → Classify → Delete ==========
def moderate_comments(video_id):
    youtube = authenticate_youtube()
    comments = fetch_comments(youtube, video_id)

    print(f"\nFetched {len(comments)} comments. Scanning for abuse...\n")
    for comment_id, comment_text in comments:
        is_abusive = preprocess_and_predict(comment_text)
        if is_abusive:
            print(f"⚠️ Abusive: {comment_text}")
            delete_comment(youtube, comment_id)
        else:
            print(f"✅ Safe: {comment_text}")

# ========== 7️⃣ Run ==========
if __name__ == "__main__":
    VIDEO_ID = "GmlsB4ccUr0"  # Replace with your actual video ID
    moderate_comments(VIDEO_ID)
