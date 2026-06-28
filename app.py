import streamlit as st
from transformers import AlbertTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.oauth2.credentials

MODEL_PATH = "model"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

REDIRECT_URI = st.secrets["redirect_uri"]

st.set_page_config(page_title="YouTube Abusive Comment Moderator", page_icon="🧹")

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
    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=False,
    )

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
st.title("YouTube Abusive Comment Moderator")

query_params = st.query_params

if "credentials" not in st.session_state:

    if "code" in query_params:
        try:
            flow = build_flow()
            flow.fetch_token(code=query_params["code"])
            st.session_state["credentials"] = credentials_to_dict(flow.credentials)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")

    else:
        flow = build_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            include_granted_scopes="true",
        )

        st.write("Sign in with the Google account linked to your YouTube channel to analyze and moderate comments.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.link_button("🔐 Sign in with Google", auth_url)

else:

    # ---------------- Header with icon-only Sign out button ---------------- #
    # header_left, header_right = st.columns([10, 1], gap="small", vertical_alignment="center")
    # with header_right:
    #     if st.button("🚪", key="sign_out_btn", help="Sign out"):
    #         del st.session_state["credentials"]
    #         st.rerun()

    # ---------------- Input + Scan button on the same line ---------------- #
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        video_id = st.text_input("Enter YouTube Video ID (the part after watch?v= in the URL)")
    with btn_col:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        scan_clicked = st.button("Fetch")

    if scan_clicked:
        if video_id.strip() == "":
            st.warning("Please enter a Video ID.")
        else:
            youtube = get_youtube_client()
            try:
                with st.spinner("Fetching comments..."):
                    comments = fetch_comments(youtube, video_id)

                results = []
                for cid, text in comments:
                    label = predict_abuse(text)
                    results.append({"id": cid, "text": text, "label": label})

                st.session_state["results"] = results

            except Exception as e:
                st.error(f"Error: {e}")

    if "results" in st.session_state:
        results = st.session_state["results"]

        safe_count = sum(r["label"] == "Safe" for r in results)
        abusive_count = sum(r["label"] == "Abusive" for r in results)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(results))
        c2.metric("Safe", safe_count)
        c3.metric("Abusive", abusive_count)

        # Only count abusive comments that haven't already been deleted
        abusive = [r for r in results if r["label"] == "Abusive" and not r.get("deleted", False)]

        youtube = get_youtube_client()

        # ---------- Delete All ---------- #
        if abusive:
            if st.button(f"🗑 Delete All Abusive Comments ({len(abusive)})"):

                deleted = 0

                for item in abusive:
                    try:
                        delete_comment(youtube, item["id"])
                        deleted += 1

                        # Mark as deleted in-place, same as individual delete —
                        # no re-fetch, no new styling, same [Deleted] block used below.
                        for r in st.session_state["results"]:
                            if r["id"] == item["id"]:
                                r["deleted"] = True
                                break
                    except Exception:
                        pass

                st.session_state["delete_all_success"] = deleted
                st.rerun()

        if "delete_all_success" in st.session_state:
            st.success(
                f"Successfully deleted {st.session_state['delete_all_success']} abusive comment(s)."
            )
            del st.session_state["delete_all_success"]

        st.subheader("🧪 Moderation Results")

        for item in results:

            left, right = st.columns([15, 1], gap="small", vertical_alignment="center")

            if item["label"] == "Abusive":

                with left:

                    if item.get("deleted", False):

                        st.markdown(
                            f"""
                            <div style="
                                background:#e9ecef;
                                border-left:6px solid #6c757d;
                                color:#444;
                                padding:15px;
                                border-radius:8px;
                                margin-bottom:10px;
                            ">
                                <b>[Deleted] </b>
                                {item['text']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:

                        st.error(f"[Abusive] {item['text']}")
                with right:

                    if not item.get("deleted", False):

                        if st.button(
                            "🗑️",
                            key=item["id"],
                            help="Delete this comment"
                        ):

                            try:
                                delete_comment(youtube, item["id"])

                                for r in st.session_state["results"]:
                                    if r["id"] == item["id"]:
                                        r["deleted"] = True
                                        break

                                st.rerun()

                            except Exception as e:
                                st.error(f"Failed to delete comment {item['id']}: {e}")
            else:

                with left:
                    st.info(f"[Safe] {item['text']}")

                with right:
                    st.empty()