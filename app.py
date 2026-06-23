import streamlit as st
from transformers import AlbertTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.oauth2.credentials

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "model"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# This MUST exactly match the Authorized redirect URI you set
# in Google Cloud Console (no trailing slash mismatch!).
REDIRECT_URI = st.secrets["redirect_uri"]

st.set_page_config(page_title="YouTube Abusive Comment Moderator", page_icon="🧹")

# ============================================================
# LOAD MODEL (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def predict_abuse(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    logits = model(**inputs).logits
    pred = np.argmax(logits, axis=1)[0]
    return "Abusive" if pred == 1 else "Safe"

# ============================================================
# OAUTH HELPERS (Web flow — works on a deployed server)
# ============================================================
def build_flow():
    """Builds the OAuth flow using client config from secrets.toml (no JSON file needed on disk)."""
    client_config = {
        "web": {
            "client_id": st.secrets["google_client"]["client_id"],
            "client_secret": st.secrets["google_client"]["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)

def credentials_to_dict(credentials):
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }

def get_youtube_client():
    if "credentials" not in st.session_state:
        return None
    creds = google.oauth2.credentials.Credentials(**st.session_state["credentials"])
    return build("youtube", "v3", credentials=creds)

# ============================================================
# YOUTUBE HELPERS
# ============================================================
def fetch_comments(youtube, video_id, max_results=50):
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
    )
    response = request.execute()
    comments = []
    for item in response.get("items", []):
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        cid = item["id"]
        comments.append((cid, text))
    return comments

def delete_comment(youtube, comment_id):
    youtube.comments().setModerationStatus(id=comment_id, moderationStatus="rejected").execute()

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🧹 YouTube Abusive Comment Auto-Moderator")

# Handle redirect back from Google (it appends ?code=... and ?state=... to the URL)
query_params = st.query_params

if "credentials" not in st.session_state:
    if "code" in query_params:
        try:
            flow = build_flow()
            st.write("Verifier:", st.session_state.get("code_verifier"))
            flow.code_verifier = st.session_state.get("code_verifier")

            flow.fetch_token(
                code=query_params["code"]
            )
            st.session_state["credentials"] = credentials_to_dict(flow.credentials)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
    else:
        flow = build_flow()

        auth_url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            include_granted_scopes="true",
        )

        st.session_state["oauth_state"] = state
        st.session_state["code_verifier"] = flow.code_verifier

        st.write("Sign in with the Google account that owns the YouTube channel/video.")
        st.link_button("🔐 Sign in with Google", auth_url)

else:
    st.success("✅ Signed in to YouTube.")
    if st.button("Sign out"):
        del st.session_state["credentials"]
        st.rerun()

    video_id = st.text_input("Enter YouTube Video ID (the part after watch?v= in the URL)")

    if st.button("Fetch, Classify & Auto-Delete Abusive Comments"):
        if video_id.strip() == "":
            st.warning("Please enter a Video ID.")
        else:
            youtube = get_youtube_client()
            try:
                with st.spinner("Fetching comments..."):
                    comments = fetch_comments(youtube, video_id)
                st.success(f"Fetched {len(comments)} comments.")

                st.subheader("🧪 Moderation Results")
                for cid, text in comments:
                    label = predict_abuse(text)
                    if label == "Abusive":
                        st.error(f"[Abusive] {text}")
                        try:
                            delete_comment(youtube, cid)
                            st.warning(f"🗑️ Deleted comment: {cid}")
                        except Exception as e:
                            st.error(f"❌ Failed to delete comment {cid}: {e}")
                    else:
                        st.info(f"[Safe] {text}")
            except Exception as e:
                st.error(f"Error: {e}")
