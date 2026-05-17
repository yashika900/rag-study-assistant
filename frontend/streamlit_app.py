"""Streamlit UI for the RAG Study Assistant."""

from __future__ import annotations

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon=":books:",
    layout="wide",
)

st.title("RAG Study Assistant")
st.caption("Upload your notes, then ask questions answered only from that material.")


# ---------------- API HELPERS ----------------


def post_chat(question: str) -> dict:
    """Send a question to the FastAPI backend."""

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"question": question},
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def upload_document(uploaded_file) -> dict:
    """Upload one file to backend."""

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=600,
    )

    response.raise_for_status()

    return response.json()


def clear_backend_chat() -> None:
    """Clear backend chat history."""

    response = requests.post(
        f"{API_BASE_URL}/clear-chat",
        timeout=30,
    )

    response.raise_for_status()
def download_chat_pdf():
    """Download exported PDF."""

    response = requests.get(
        f"{API_BASE_URL}/export-pdf",
        timeout=60,
    )

    response.raise_for_status()

    return response.content


def get_error_message(exc: requests.HTTPError, fallback: str) -> str:
    """Extract useful backend error message."""

    try:
        payload = exc.response.json()

    except ValueError:
        return fallback

    detail = payload.get("detail")

    if isinstance(detail, dict):
        return detail.get("error") or str(detail)

    if detail:
        return str(detail)

    return str(payload.get("error") or fallback)


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.divider()

    st.header("Study Material")

    uploaded_files = st.file_uploader(
         "Upload PDF, TXT, DOCX, or Images",
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    upload_clicked = st.button(
        "Upload and Index",
        type="primary",
        use_container_width=True,
    )

    if upload_clicked:

        if not uploaded_files:

            st.warning("Choose at least one file first.")

        else:

            with st.spinner("Reading and indexing documents..."):

                successful_uploads = 0

                for uploaded_file in uploaded_files:

                    try:

                        result = upload_document(uploaded_file)

                        # Avoid duplicates in sidebar
                        if uploaded_file.name not in st.session_state.uploaded_docs:
                            st.session_state.uploaded_docs.append(
                                uploaded_file.name
                            )

                        successful_uploads += 1

                        st.success(
                            f"{result['filename']} indexed with "
                            f"{result['chunks_created']} chunks."
                        )

                    except requests.HTTPError as exc:

                        st.error(
                            f"{uploaded_file.name}: "
                            f"{get_error_message(exc, 'Upload failed.')}"
                        )

                    except requests.RequestException as e:

                        st.error(
                            f"{uploaded_file.name}: REQUEST ERROR: {str(e)}"
                        )

                        print("REQUEST ERROR:", str(e))

                if successful_uploads > 0:

                    st.success(
                        f"Successfully indexed "
                        f"{successful_uploads} document(s)."
                    )

    # ---------- Uploaded Docs ----------

    if st.session_state.uploaded_docs:

        st.subheader("Uploaded Documents")

        for doc_name in st.session_state.uploaded_docs:

            st.markdown(f"- {doc_name}")

    st.divider()

    # ---------- Export PDF ----------

    if st.button(
        "Export Chat as PDF",
        use_container_width=True,
    ):

        try:

            pdf_bytes = download_chat_pdf()

            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="study_assistant_chat.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        except requests.RequestException:

            st.error("Could not export PDF.")

    st.divider()

    # ---------- Clear Chat ----------

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):

        try:

            clear_backend_chat()

            st.session_state.messages = []

            st.success("Chat cleared.")

        except requests.RequestException:

            st.error("Could not clear chat on the backend.")

    st.caption(f"Backend: {API_BASE_URL}")


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Sources"):

                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):

                    page = (
                        f" | page {source['page']}"
                        if source.get("page")
                        else ""
                    )

                    st.markdown(
                        f"**{index}. {source['source']}{page}**"
                    )

                    st.write(source["content"])


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask a question from the uploaded material"
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching your material..."):

            try:

                result = post_chat(question)

                st.markdown(result["answer"])

                if result.get("sources"):

                    with st.expander("Sources"):

                        for index, source in enumerate(
                            result["sources"],
                            start=1,
                        ):

                            page = (
                                f" | page {source['page']}"
                                if source.get("page")
                                else ""
                            )

                            st.markdown(
                                f"**{index}. {source['source']}{page}**"
                            )

                            st.write(source["content"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )

            except requests.HTTPError as exc:

                st.error(
                    get_error_message(exc, "Chat failed.")
                )

            except requests.RequestException as e:

                st.error(
                    f"REQUEST ERROR: {str(e)}"
                )

                print("REQUEST ERROR:", str(e))