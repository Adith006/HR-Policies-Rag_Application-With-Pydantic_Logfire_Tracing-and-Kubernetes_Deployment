
import os
import uuid

import httpx
import streamlit as st

st.set_page_config(
    page_title="HR Policies Assistant",
    page_icon="📚",
    layout="wide",
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")


@st.cache_resource
def get_api_client() -> httpx.Client:
    """Reuse the HTTP connection pool across Streamlit reruns."""
    return httpx.Client(
        headers={"Accept": "application/json"},
        timeout=httpx.Timeout(180.0, connect=5.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
        follow_redirects=True,
    )


api_client = get_api_client()

st.title("Enterprise AI Knowledge Platform")
st.caption("Ask questions about your HR policies and inspect the sources.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def display_sources(sources):
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            similarity = source.get("similarity")
            score = (
                f"{float(similarity):.3f}"
                if similarity is not None
                else "N/A"
            )

            st.markdown(
                f"**{source.get('source', 'Unknown source')}** "
                f"— Page {source.get('page_number', 'N/A')}"
            )
            st.caption(
                f"Section: {source.get('section_title', 'Unknown')} "
                f"| Similarity: {score}"
            )


def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            cache_status = message.get("cache")
            if cache_status:
                st.caption(f"Cache: {cache_status}")

            display_sources(message.get("sources", []))


with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL, language=None)

    if st.button("Check API health", use_container_width=True):
        try:
            response = api_client.get(
                f"{API_BASE_URL}/health",
                timeout=5,
            )
            response.raise_for_status()
            st.success("API is healthy.")
        except httpx.HTTPError as exc:
            st.error(f"Cannot reach the API: {exc}")

    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.caption("Conversation ID")
    st.code(st.session_state.session_id, language=None)


for message in st.session_state.messages:
    display_message(message)


question = st.chat_input("Ask a question about your HR policies…")

if question:
    user_message = {
        "role": "user",
        "content": question,
    }
    st.session_state.messages.append(user_message)
    display_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating an answer…"):
            try:
                response = api_client.post(
                    f"{API_BASE_URL}/ask",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "user_id": st.session_state.user_id,
                    },
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    st.error("The API returned an empty answer.")
                    st.stop()

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": data.get("sources", []),
                    "cache": data.get("cache"),
                }

                st.markdown(answer)

                if assistant_message["cache"]:
                    st.caption(
                        f"Cache: {assistant_message['cache']}"
                    )

                display_sources(assistant_message["sources"])
                st.session_state.messages.append(assistant_message)

            except httpx.TimeoutException:
                st.error("The request timed out. Please try again.")
            except httpx.HTTPStatusError as exc:
                st.error(f"The API returned HTTP {exc.response.status_code}.")
            except httpx.RequestError:
                st.error(
                    "Could not connect to the API. "
                    "Check the API URL and service status."
                )
            except ValueError:
                st.error("The API returned an invalid JSON response.")
