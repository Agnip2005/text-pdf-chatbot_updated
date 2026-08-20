"""
Streamlit demo UI.

This is now a thin client on top of the FastAPI backend (api.py). The user
uploads any PDF or TXT file(s) from the sidebar; the app sends them to the
API to be indexed (Qdrant + BM25 hybrid + BGE reranker), and the chat below
answers questions using only those uploaded file(s). Uploading new files
replaces the previous ones for that browser session.

Run the API first:
    uvicorn api:app --reload --port 8000
Then run this app:
    streamlit run app.py
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def safe_error_detail(resp: requests.Response) -> str:
    """
    Best-effort extraction of an error message from a non-2xx response.
    The API is supposed to always return JSON, but if it crashed hard
    (or a proxy/host returned an HTML error page, or the server simply
    isn't running the code we think it is), resp.json() itself can raise
    requests.exceptions.JSONDecodeError. Never let that bubble up and
    crash the whole Streamlit app - fall back to the raw response text.
    """
    try:
        data = resp.json()
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
        return str(data)
    except ValueError:
        text = resp.text.strip()
        return text if text else f"HTTP {resp.status_code} with an empty response body."

st.set_page_config(page_title="Document Chatbot", page_icon="📄")
st.title("📄 Document Chatbot")
st.caption("Upload a PDF or TXT file, then ask questions about it.")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "files_ready" not in st.session_state:
    st.session_state.files_ready = False

if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []


def ensure_session():
    if st.session_state.session_id is None:
        resp = requests.post(f"{API_BASE_URL}/session")
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Sidebar: file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload document(s)")
    st.write("Every new upload replaces the previous file(s) for this session.")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT file(s)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process file(s)", disabled=not uploaded_files):
        try:
            ensure_session()
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in uploaded_files
            ]
            with st.spinner("Indexing document(s)... (chunking, embedding, hybrid index)"):
                resp = requests.post(
                    f"{API_BASE_URL}/upload/{st.session_state.session_id}",
                    files=files_payload,
                )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.files_ready = True
                st.session_state.ingested_sources = data["files_ingested"]
                st.session_state.messages = []  # fresh chat for the new document(s)
                st.success(
                    f"Indexed {len(data['files_ingested'])} file(s), "
                    f"{data['chunk_count']} chunks."
                )
            else:
                st.error(f"Upload failed: {safe_error_detail(resp)}")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Couldn't reach the API at {API_BASE_URL}. "
                "Make sure it's running: `uvicorn api:app --reload --port 8000`"
            )
        except Exception as e:
            st.error(f"Unexpected error while uploading: {e}")

    if st.session_state.files_ready:
        st.divider()
        st.subheader("Active document(s)")
        for src in st.session_state.ingested_sources:
            st.write(f"- {src}")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
if not st.session_state.files_ready:
    st.info("Upload a PDF or TXT file from the sidebar and click **Process file(s)** to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about your uploaded document(s)")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/ask/{st.session_state.session_id}",
                        json={"question": question},
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["answer"]
                    else:
                        answer = f"Error: {safe_error_detail(resp)}"
                except requests.exceptions.ConnectionError:
                    answer = (
                        f"Couldn't reach the API at {API_BASE_URL}. "
                        "Make sure it's running: `uvicorn api:app --reload --port 8000`"
                    )
                except Exception as e:
                    answer = f"Unexpected error while getting a response: {e}"
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
