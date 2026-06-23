"""
deletion.py
Command-line version of the moderation logic, using the SAME web-based
OAuth flow as app.py (instead of InstalledAppFlow.run_local_server()).

This is useful to test your model + YouTube logic from the terminal
without going through Streamlit. It still uses a "Sign in with Google"
style flow: it prints a URL, you open it in a browser, log in, and paste
the resulting code back into the terminal.

For the actual deployed product, app.py is what real users interact with.
This file is for YOUR OWN testing/debugging.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import tensorflow as tf
from transformers import AlbertTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.oauth2.credentials

# ========== 1. Load Model and Tokenizer ==========
MODEL_PATH = "model"

print("Loading model and tokenizer...")
tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print("Model and tokenizer loaded ✅")

# ========== 2. Classify Comment ==========
def preprocess_and_predict(text):
    """Returns True if the comment is classified as abusive."""
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    logits = model(**inputs).logits
    prediction = np.argmax(logits, axis=1)[0]  # 0 = safe, 1 = abusive
    return prediction == 1

# ========== 3. YouTube Authentication (Web flow, no local server) ==========
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# Use the SAME web client_id/secret you created in Google Cloud Console
# (the one you'll also put into Streamlit secrets for app.py).
# For local CLI testing, you can either:
#   (a) paste a client_secret.json (web type) path below, or
#   (b) hardcode client_id/client_secret directly.
CLIENT_SECRET_FILE = "client_secret_web.json"  # the WEB client JSON, not desktop type

# For local CLI testing only — Google allows "http://localhost" as a
# redirect URI on a Web client if you add it in Cloud Console.
REDIRECT_URI = "http://localhost:8080/"

def authenticate_youtube():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")

    print("\n1. Open this URL in your browser and log in:")
    print(auth_url)
    print("\n2. After approving, you'll be redirected to a localhost URL that")
    print("   will fail to load (that's expected) — copy the FULL URL from")
    print("   your browser's address bar and paste it below.\n")

    redirected_url = input("Paste the full redirected URL here: ").strip()

    # Extract the 'code' param from the pasted URL
    from urllib.parse import urlparse, parse_qs
    code = parse_qs(urlparse(redirected_url).query).get("code", [None])[0]
    if not code:
        raise ValueError("Could not find 'code' in the pasted URL.")

    flow.fetch_token(code=code)
    return build("youtube", "v3", credentials=flow.credentials)

# ========== 4. Fetch Comments ==========
def fetch_comments(youtube, video_id, max_results=100):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText",
    )
    response = request.execute()

    comments = []
    for item in response.get("items", []):
        comment_id = item["id"]
        comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append((comment_id, comment_text))
    return comments

# ========== 5. Delete Comment ==========
def delete_comment(youtube, comment_id):
    request = youtube.comments().setModerationStatus(
        id=comment_id,
        moderationStatus="rejected",
        banAuthor=False,
    )
    request.execute()
    print(f"❌ Deleted abusive comment: {comment_id}")

# ========== 6. Combine: Fetch → Classify → Delete ==========
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

# ========== 7. Run ==========
if __name__ == "__main__":
    VIDEO_ID = input("Enter the YouTube Video ID: ").strip()
    moderate_comments(VIDEO_ID)
