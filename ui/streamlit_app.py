

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

import jwt
import streamlit as st
from dotenv import load_dotenv

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load local environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Load Streamlit Cloud secrets into environment variables
SECRET_KEYS = (
    "DATABASE_URL",
    "STREAMLIT_AUTH_SECRET",
    "EVALUATION_API_TOKEN",
)

for key in SECRET_KEYS:
    if not os.getenv(key) and key in st.secrets:
        os.environ[key] = str(st.secrets[key])

# Load Streamlit Cloud secrets
if not os.getenv("DATABASE_URL"):
    if "DATABASE_URL" in st.secrets:
        os.environ["DATABASE_URL"] = str(st.secrets["DATABASE_URL"])

# Validate configuration before importing application modules
if not os.getenv("DATABASE_URL"):
    raise RuntimeError(
        "DATABASE_URL is missing. Configure it in Streamlit Cloud Secrets."
    )

# Application imports must come AFTER configuration
from app.conversation_store import get_messages


# Safe diagnostic: never print the actual database URL
st.write(
    "Database configuration loaded:",
    bool(os.getenv("DATABASE_URL"))
)

# Only now import application modules
import jwt
from datetime import datetime, timedelta, timezone
# Diagnose Streamlit database configuration
try:
    secret_exists = "DATABASE_URL" in st.secrets

    if secret_exists and not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = str(st.secrets["DATABASE_URL"])

    st.write("Streamlit secret exists:", secret_exists)
    st.write("Environment variable loaded:", bool(os.getenv("DATABASE_URL")))

except Exception as exc:
    st.error(f"Secret loading failed: {type(exc).__name__}: {exc}")

from app.conversation_store import get_messages

# Import application modules only after setting the project path.
from ui.HR_Policy_Manual import display_policy_manual


st.set_page_config(
    page_title="Enterprise AI Knowledge Platform",
    page_icon="📚",
    layout="wide",
)

# ==========================================
# GOOGLE AUTHENTICATION
# ==========================================

if not st.user.is_logged_in:
    st.title("HR Policies Assistant")
    st.write("Sign in with Google to access the HR Policies Assistant.")

    if st.button("Continue with Google", type="primary"):
        st.login()

    st.stop()



google_user_id = st.user.get("sub")
google_email_verified = st.user.get("email_verified", False)

if not google_user_id or google_email_verified is not True:
    st.error("Authentication failed. Please sign in again.")
    st.stop()

CHAT_USER_ID = str(google_user_id)



# Style the Ragas panel as a right-hand sidebar.
st.markdown(
    """
    <style>
    .st-key-ragas-right-sidebar {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 0.75rem;
        padding: 1rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG_KEYS = [
    "DATABASE_URL",
    "REDIS_URL",
    "PORTKEY_API_KEY",
    "GROQ_MODEL",
    "OPENAI_MODEL",
    "LOGFIRE_TOKEN",
    "API_BASE_URL",
     "EVALUATION_API_TOKEN",
]

# Streamlit Cloud secrets override local environment values.
for key in CONFIG_KEYS:
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None

    if value:
        os.environ[key] = str(value)

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_MODE = bool(API_BASE_URL)
print("APPLICATION MODE:", "API" if API_MODE else "DIRECT")
print("API BASE URL CONFIGURED:", bool(API_BASE_URL))

if not API_MODE:
    required_keys = [
        "DATABASE_URL",
        "REDIS_URL",
        "PORTKEY_API_KEY",
        "GROQ_MODEL",
        "OPENAI_MODEL",
    ]

    missing_keys = [
        key for key in required_keys
        if not os.getenv(key)
    ]

    if missing_keys:
        st.error(
            "Missing application configuration: "
            + ", ".join(missing_keys)
        )
        st.stop()

CHAT_INPUT_KEY = "hr_policy_chat_input"


def select_sample_question(question: str) -> None:
    """
    Populate the chat input when a sample question is clicked.
    The question is not submitted automatically.
    """
    st.session_state[CHAT_INPUT_KEY] = question


def display_sources(sources):
    """Display source citations for a normal chat response."""
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
                f"- Page {source.get('page_number', 'N/A')}"
            )

            st.caption(
                f"Section: {source.get('section_title', 'Unknown')} "
                f"| Similarity: {score}"
            )


def display_trace(events):
    """Display captured stages of a normal RAG chat request."""
    if not events:
        return

    completed = sum(
        event.get("status") == "completed"
        for event in events
    )

    denied = any(
        event.get("status") == "blocked"
        for event in events
    )

    failed = sum(
        event.get("status") == "failed"
        for event in events
    )

    with st.expander(
        f"How this answer was produced - "
        f"{completed}/{len(events)} stages completed"
    ):
        for event in events:
            status = event.get("status", "unknown")
            stage = event.get("stage", "Unknown stage")
            duration = event.get("duration_ms", 0)

            st.markdown(
                f"**- {stage}** - "
                f"{status.capitalize()} - "
                f"{duration:.2f} ms"
            )

            details = []

            for key in (
                "model",
                "candidate_count",
                "result_count",
                "context_chunk_count",
                "reason",
                "top_score",
            ):
                value = event.get(key)

                if value is not None:
                    details.append(
                        f"{key.replace('_', ' ').title()}: {value}"
                    )

            if details:
                st.caption(" - ".join(details))

        if denied:
            st.info(
                "Request denied by guardrails. "
                "Cache bypassed and answer generation skipped."
            )
        elif failed:
            st.error(f"{failed} pipeline stage(s) failed.")

def display_retrieved_contexts(contexts):
    """
    Display the actual chunks returned by the RAG retriever.
    Supports strings, dictionaries and LangChain Document objects.
    """

    if not contexts:
        return

    if isinstance(contexts, (str, dict)) or hasattr(contexts, "page_content"):
        contexts = [contexts]

    with st.expander(
        f"Retrieved Contexts ({len(contexts)})",
        expanded=False,
    ):

        for index, item in enumerate(contexts, start=1):

            metadata = {}

            if isinstance(item, dict):

                metadata = item.get("metadata") or {}

                context_text = (
                    item.get("text")
                    or item.get("page_content")
                    or item.get("content")
                    or item.get("chunk_text")
                    or ""
                )

                if not context_text and "document" in item:
                    document = item["document"]
                    context_text = getattr(
                        document,
                        "page_content",
                        str(document),
                    )

            elif hasattr(item, "page_content"):

                context_text = item.page_content
                metadata = getattr(item, "metadata", {}) or {}

            else:

                context_text = str(item)

            source = (
                item.get("source") if isinstance(item, dict) else None
            ) or metadata.get("source")

            page = (
                item.get("page_number") if isinstance(item, dict) else None
            )

            if page is None:
                page = metadata.get("page_number")

            section = (
                item.get("section_title") if isinstance(item, dict) else None
            )

            if section is None:
                section = metadata.get("section_title")

            similarity = (
                item.get("similarity") if isinstance(item, dict) else None
            )

            if similarity is None:
                similarity = metadata.get("similarity")

            details = []

            if source:
                details.append(f"Source: {source}")

            if page is not None:
                details.append(f"Page: {page}")

            if section:
                details.append(f"Section: {section}")

            if similarity is not None:
                details.append(f"Similarity: {similarity}")

            st.markdown(f"**Context {index}**")

            if details:
                st.caption(" | ".join(map(str, details)))

            if context_text:
                st.markdown(context_text)
            else:
                st.write(item)

            if index < len(contexts):
                st.divider()

def display_assistant_details(message: dict) -> None:
    """
    Display all additional information associated with an answer.
    """

    # Cache status
    cache = message.get("cache")

    if cache is not None:
        st.caption(f"Cache: {cache}")

    # Retrieved document sources
    sources = message.get("sources", [])

    if sources:
        display_sources(sources)

    # Actual retrieved chunks
    contexts = message.get("retrieved_contexts", [])

    if contexts:
        display_retrieved_contexts(contexts)

    # RAG pipeline stages
    trace = message.get("trace", [])
    if trace:
        display_trace(trace)

def display_message(message: dict) -> None:
    """
    Display a user or assistant message from conversation history.
    """

    role = message.get("role", "assistant")

    with st.chat_message(role):

        st.markdown(message.get("content", ""))

        if role == "assistant":
            display_assistant_details(message)



@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """Configure Logfire and reuse the RAG pipeline."""

    import logfire

    logfire_token = os.getenv("LOGFIRE_TOKEN")

    if not logfire_token:
        raise RuntimeError("LOGFIRE_TOKEN is missing.")

    logfire.configure(
        token=logfire_token,
        send_to_logfire=True,
        service_name="enterprise-ai-knowledge-platform-streamlit",
        environment="development",
        inspect_arguments=False,
        console=False,
    )

    from app.rag_pipeline import answer_question
    from app.rag_trace import capture_rag_trace

    logfire.info("Streamlit RAG pipeline initialized")

    return answer_question, capture_rag_trace



@st.cache_resource(show_spinner=False)
def get_api_client():
    """Create and reuse the FastAPI HTTP client."""
    import httpx

    return httpx.Client(
        headers={"Accept": "application/json"},
        timeout=httpx.Timeout(180.0, connect=5.0),
        follow_redirects=True,
    )


# @st.cache_resource(show_spinner=False)
# def get_conversation_store():
#     """Load persistent conversation storage functions."""
#     from app.conversation_store import (
#         create_conversation,
#         save_message,
#         get_conversations,
#         get_messages,
#         set_conversation_pinned,
#         delete_conversation,
#     )

#     return {
#         "create": create_conversation,
#         "save": save_message,
#         "list": get_conversations,
#         "messages": get_messages,
#         "pin": set_conversation_pinned,
#         "delete": delete_conversation,
#     }

def get_api_headers():
    """Create authentication headers for FastAPI."""
    auth_now = datetime.now(timezone.utc)

    token = jwt.encode(
        {
            "sub": CHAT_USER_ID,
            "email_verified": True,
            "iss": "enterprise-hr-assistant",
            "aud": "enterprise-hr-api",
            "iat": auth_now,
            "exp": auth_now + timedelta(seconds=90),
        },
        st.secrets["STREAMLIT_AUTH_SECRET"],
        algorithm="HS256",
    )

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

def api_request(method, endpoint, **kwargs):
    """Send an authenticated request to FastAPI."""

    response = get_api_client().request(
        method,
        f"{API_BASE_URL}{endpoint}",
        headers=get_api_headers(),
        **kwargs,
    )

    response.raise_for_status()

    if response.content:
        return response.json()

    return {}


@st.cache_data(ttl=30, max_entries=20, show_spinner=False)
def get_recent_conversations(user_id: str, limit: int = 50):
    """Retrieve conversation history from the active application mode."""

    if API_MODE:
        data = api_request(
            "GET",
            "/conversations",
            params={"limit": limit},
        )

        return data.get("conversations", [])

    # Use the local database in direct development mode.
    from app.conversation_store import get_conversations

    return get_conversations(
        user_id=user_id,
        limit=limit,
    )


@st.cache_resource(show_spinner=False)
def get_short_term_memory():
    """Reuse the Redis conversation memory client."""
    from app.memory import ConversationMemory

    return ConversationMemory()


def start_new_conversation():
    """Start a fresh chat without deleting the previous conversation."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.conversation_initialized = False


def persist_conversation_message(message):
    """Save messages locally only when running outside API mode."""

    if API_MODE:
        return

    from app.conversation_store import (
        create_conversation,
        save_message,
    )

    conversation_id = st.session_state.session_id
    user_id = st.session_state.user_id

    # Ensure the conversation exists before saving its message.
    create_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )

    metadata = {}

    if message["role"] == "assistant":
        metadata = {
            "sources": message.get("sources", []),
            "cache": message.get("cache"),
            "trace": message.get("trace", []),
            "retrieved_contexts": message.get(
                "retrieved_contexts", []
            ),
        }

    save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role=message["role"],
        content=message["content"],
        metadata=metadata,
    )

    get_recent_conversations.clear()

def load_saved_conversation(conversation_id):
    """Load a saved conversation using the active application mode."""

    if API_MODE:
        # Load messages through FastAPI.
        data = api_request(
            "GET",
            f"/conversations/{conversation_id}/messages",
        )
        saved_messages = data.get("messages", [])

    else:
        # Load messages directly from the local database.
        from app.conversation_store import get_messages

        saved_messages = get_messages(
            conversation_id=conversation_id,
            user_id=st.session_state.user_id,
        )

    st.session_state.session_id = conversation_id
    st.session_state.messages = []

    for message in saved_messages:
        metadata = message.get("metadata") or {}

        restored_message = {
            "role": message["role"],
            "content": message["content"],
        }

        if message["role"] == "assistant":
            restored_message.update({
                "sources": metadata.get("sources", []),
                "cache": metadata.get("cache"),
                "trace": metadata.get("trace", []),
                "retrieved_contexts": metadata.get(
                    "retrieved_contexts", []
                ),
            })

        st.session_state.messages.append(restored_message)

    st.session_state.conversation_initialized = True
    st.rerun()

def toggle_conversation_pin(conversation):
    """Pin or unpin using the active application mode."""

    new_pin_state = not conversation["is_pinned"]

    try:
        if API_MODE:
            # Update through FastAPI.
            api_request(
                "PATCH",
                f"/conversations/{conversation['id']}/pin",
                json={"is_pinned": new_pin_state},
            )

        else:
            # Update directly in the local database.
            from app.conversation_store import set_conversation_pinned

            updated = set_conversation_pinned(
                conversation_id=conversation["id"],
                user_id=st.session_state.user_id,
                is_pinned=new_pin_state,
            )

            if not updated:
                raise RuntimeError("Conversation was not found.")

        # Refresh the sidebar history.
        get_recent_conversations.clear()

    except Exception as exc:
        st.error(f"Unable to update conversation pin: {exc}")

    st.rerun()


def remove_saved_conversation(conversation_id):
    """Delete a conversation using the active application mode."""

    try:
        if API_MODE:
            # Delete through FastAPI.
            api_request(
                "DELETE",
                f"/conversations/{conversation_id}",
            )
        else:
            # Delete directly in local development mode.
            from app.conversation_store import delete_conversation

            deleted = delete_conversation(
                conversation_id=conversation_id,
                user_id=st.session_state.user_id,
            )

            if not deleted:
                raise RuntimeError("Conversation was not found.")

            # Clear Redis history in direct mode.
            from app.memory import ConversationMemory

            memory = ConversationMemory()
            memory.clear_history(
                st.session_state.user_id,
                conversation_id,
            )

        get_recent_conversations.clear()

        if st.session_state.session_id == conversation_id:
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.conversation_initialized = False

        st.session_state.pop("pending_delete_id", None)
        st.rerun()

    except Exception as exc:
        st.error(f"Unable to delete conversation: {exc}")



def run_chat_question(question):
    """Run chat through FastAPI or direct local RAG mode."""
    if API_MODE:

        auth_now = datetime.now(timezone.utc)
        google_id_token = jwt.encode(
            {
                "sub": CHAT_USER_ID,
                "email_verified": True,
                "iss": "enterprise-hr-assistant",
                "aud": "enterprise-hr-api",
                "iat": auth_now,
                "exp": auth_now + timedelta(seconds=90),
                },
                st.secrets["STREAMLIT_AUTH_SECRET"],
                algorithm="HS256",
                )
        client = get_api_client()
        response = client.post(
            f"{API_BASE_URL}/ask",
            headers={
                "Authorization": f"Bearer {google_id_token}",
                },
                json={
                    "question": question,
                    "session_id": st.session_state.session_id,
                    },
                )



        response.raise_for_status()
        data = response.json()
        trace_events = data.get("trace", [])

        return data, trace_events


    import logfire

    answer_question, capture_rag_trace = load_rag_pipeline()

    # Parent span groups this question's RAG operations in Logfire.
    with logfire.span("streamlit.rag_request") as request_span:
        request_span.set_attribute("application.mode", "streamlit_direct")

        with capture_rag_trace() as trace_events:
            result = answer_question(
                question=question,
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_id,
            )

        request_span.set_attribute(
            "rag.trace_stage_count",
            len(trace_events),
        )

    return result, trace_events


SAMPLE_QUESTIONS = [
    "What is the vision of the organization as stated in the policy manual?",
    "How often is payroll administered and on what dates?",
    "What percentage of the health and dental insurance premium cost does the company pay?",
    "What minimum percentage of its planned operating budget does the organization set aside to fund professional development activities?",
    "What is the maximum duration specified for daily standup meetings?",
]

METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer relevancy"),
    ("context_precision", "Context precision"),
    ("context_recall", "Context recall"),
]

RAGAS_CACHE_TTL_SECONDS = 3600
RAG_CORPUS_VERSION = "manual.pdf-58-chunks-v1"

EVALUATOR_VERSION = (
    "ragas-0.3.1-question-cards-f1-reference-similarity-v3"
)


@st.cache_data(
    ttl=RAGAS_CACHE_TTL_SECONDS,
    max_entries=5,
    show_spinner=False,
)
def cached_ragas_evaluation(
    question_count,
    dataset_hash,
    model_name,
    corpus_version,
    evaluator_version,
):
    """Cache identical evaluations returned by the FastAPI backend."""

    if API_MODE:
        token = os.getenv("EVALUATION_API_TOKEN")

        if not token:
            raise RuntimeError(
                "EVALUATION_API_TOKEN is missing."
            )

        client = get_api_client()

        response = client.post(
            f"{API_BASE_URL}/evaluate",
            json={"question_count": question_count},
            headers={
                "X-Evaluation-Token": token,
            },
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "completed":
            raise RuntimeError(
                "FastAPI evaluation did not complete successfully."
            )

        # The backend wraps the evaluation report inside "results".
        return payload["results"]

    # Preserve local direct execution mode for development.
    from evals.run_ragas import run_evaluation

    return run_evaluation(question_count)


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def load_golden_samples(dataset_mtime_ns: int):
    """Load and cache the five golden evaluation questions."""
    dataset_path = PROJECT_ROOT / "evals" / "dataset.json"

    with dataset_path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list) or len(samples) != 5:
        raise ValueError(
            "Expected exactly five golden evaluation samples."
        )

    return samples



def format_evaluation_score(value):
    """Format a numeric evaluation score safely."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def display_question_evaluation(item, display_number):
    """Display one question's evaluation as a vertical card."""
    question_number = item.get("question_number", display_number)

    with st.container(border=True):
        st.markdown(f"### Question {question_number}")

        # Question
        st.markdown("**Question**")
        st.write(item.get("question", "Unavailable"))

        st.divider()

        # Generated answer
        st.markdown("**Generated output — RAG**")
        st.write(
            item.get("generated_answer")
            or "Generated answer unavailable."
        )

        st.divider()

        # Golden reference
        st.markdown(
            "**Actual output — Golden reference (`dataset.json`)**"
        )
        st.write(
            item.get("reference_answer")
            or "Golden reference unavailable."
        )

        st.divider()

        # Question-wise cumulative average
        st.markdown("**Question-wise cumulative average**")
        st.metric(
            "Average of the four Ragas metrics",
            format_evaluation_score(
                item.get("cumulative_average")
            ),
        )

        # Four Ragas metrics
        st.markdown("**Ragas evaluation metrics**")
        scores = item.get("scores") or {}

        for metric_key, metric_label in METRICS:
            st.write(
                f"**{metric_label}:** "
                f"{format_evaluation_score(scores.get(metric_key))}"
            )

        st.divider()

        # Retrieval F1
        st.markdown("**Retrieval F1**")
        st.metric(
            "Harmonic mean of context precision and recall",
            format_evaluation_score(item.get("retrieval_f1")),
        )

        # Generated answer vs golden answer similarity
        st.markdown(
            "**Generated answer vs golden answer similarity**"
        )
        st.metric(
            "Embedding cosine similarity",
            format_evaluation_score(
                item.get("answer_reference_similarity")
            ),
        )

        st.caption(
            "Supplementary semantic similarity; this is not a "
            "factual-correctness score."
        )

        st.divider()

        # Actual retrieved document evidence
        st.markdown("**Retrieved document evidence**")

        contexts = item.get("retrieved_contexts") or []
        sources = item.get("sources") or []

        if not contexts:
            st.warning(
                "No retrieved contexts were included in this report."
            )
        else:
            for index, context in enumerate(contexts):
                source = (
                    sources[index]
                    if index < len(sources)
                    else {}
                )

                st.markdown(
                    f"**Source:** {source.get('source', 'Unknown')}"
                )

                st.caption(
                    f"Page: {source.get('page_number', 'N/A')} "
                    f"| Section: {source.get('section_title', 'Unknown')} "
                    f"| Retrieval similarity: "
                    f"{format_evaluation_score(source.get('similarity'))}"
                )

                st.text(context)

                if index < len(contexts) - 1:
                    st.divider()


def display_ragas_dashboard():
    """Render the selected-question Ragas evaluation dashboard."""
    st.subheader("Evaluation Metrics")

    st.caption(
        "Compare generated answers with golden references and inspect "
        "the document evidence retrieved by the pipeline."
    )


    question_count = st.slider(
        "Number of golden questions",
        min_value=1,
        max_value=5,
        value=5,
        step=1,
        key="ragas_question_count",
    )
    dataset_path = PROJECT_ROOT / "evals" / "dataset.json"
    selected_samples = load_golden_samples(
        dataset_path.stat().st_mtime_ns
    )[:question_count]

    # Display selection count only once
    st.caption(f"Selected questions: {question_count}/5")

    st.markdown("**Questions included in this evaluation**")

    # Display only the selected questions
    for index, sample in enumerate(selected_samples, start=1):
        st.markdown(f"**Q{index}.** {sample['question']}")



    # Run evaluation
    if st.button(
        "Run Evaluation",
        type="primary",
        use_container_width=True,
        key="run_ragas",
    ):
        st.session_state.pop("ragas_error", None)

        try:
            with st.spinner(
                f"Evaluating {question_count} selected questions "
                "across four Ragas metrics..."
            ):
                dataset_path = PROJECT_ROOT / "evals" / "dataset.json"

                dataset_hash = hashlib.sha256(
                    dataset_path.read_bytes()
                ).hexdigest()

                report = cached_ragas_evaluation(
                    question_count=question_count,
                    dataset_hash=dataset_hash,
                    model_name=os.getenv("GROQ_MODEL", ""),
                    corpus_version=RAG_CORPUS_VERSION,
                    evaluator_version=EVALUATOR_VERSION,
                )

            st.session_state.ragas_report = report

        except Exception as exc:
            import traceback

            traceback.print_exc()
            st.session_state.ragas_error = type(exc).__name__
            st.session_state.pop("ragas_report", None)

    # Clear cached evaluation
    if st.button(
        "Clear cached evaluations",
        use_container_width=True,
        key="clear_ragas_cache",
    ):
        cached_ragas_evaluation.clear()
        st.session_state.pop("ragas_report", None)
        st.session_state.pop("ragas_error", None)
        st.rerun()

    st.caption(
        "Identical evaluations are cached for up to one hour. "
        "Clear the cache to force a fresh evaluation."
    )

    # Display errors
    if st.session_state.get("ragas_error"):
        st.error(
            f"Evaluation failed: {st.session_state.ragas_error}. "
            "Check the application logs."
        )

    report = st.session_state.get("ragas_report")

    if not report:
        st.info(
            "Select 1–5 golden questions and run an evaluation "
            "to display the results."
        )
        return

    # Do not show a report generated for a different selection.
    if report.get("question_count") != question_count:
        st.info(
            f"The displayed report contains "
            f"{report.get('question_count')} questions. "
            f"Run the evaluation again for {question_count} "
            "selected questions."
        )
        return

    question_scores = report.get("question_scores") or []

    if len(question_scores) != question_count:
        st.warning(
            "The evaluation report does not contain exactly the "
            "selected number of question results."
        )
        return

    st.divider()

    st.caption(
        f"Evaluated questions: {report['question_count']}"
    )

    st.caption(
        f"Evaluated at: {report['evaluated_at']}"
    )

    # Overall cumulative average
    st.markdown("### Overall cumulative average")

    st.metric(
        "Average across selected questions",
        format_evaluation_score(
            report.get("cumulative_average")
        ),
    )

    st.caption(
        "Average of the selected questions' individual cumulative "
        "averages. Each question average uses all four Ragas metrics."
    )

    st.divider()

    # Vertical question cards
    st.markdown("### Question-wise evaluation")

    for display_number, item in enumerate(
        question_scores,
        start=1,
    ):
        display_question_evaluation(item, display_number)


# Session state
# Session state initialization

# Stable owner ID for the private single-user workspace.

def open_policy_manual():
    st.session_state.active_view = "manual"
    st.rerun()

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

st.session_state.user_id = CHAT_USER_ID

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_initialized" not in st.session_state:
    st.session_state.conversation_initialized = False

if "pending_delete_id" not in st.session_state:
    st.session_state.pending_delete_id = None

if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("HR Policies Assistant")

st.caption(
    "Document-grounded HR policy assistant with "
    "retrieval transparency and RAG evaluation."
)


# ==========================================
# LEFT NATIVE SIDEBAR
# IMPORTANT: KEEP THIS OUTSIDE st.columns()
# ==========================================

with st.sidebar:
     # ==========================================
    # AUTHENTICATED USER
    # ==========================================

    st.caption("SIGNED IN AS")

    google_user_name = st.user.get("name", "Google User")
    google_user_email = st.user.get("email", "")

    st.markdown(f"**{google_user_name}**")

    if google_user_email:
        st.caption(google_user_email)

    if st.button(
        "Sign out",
        icon=":material/logout:",
        use_container_width=True,
        key="google_sign_out",
    ):
        st.logout()

    st.divider()

    # Sidebar contents
    st.subheader("Contents")

    if st.button(
    "HR Policy Manual",
    icon=":material/menu_book:",
    use_container_width=True,
    key="open_policy_manual",
    ):
        open_policy_manual()



    st.markdown(
        "[Logfire Dashboard](https://logfire-us.pydantic.dev/adith2015/hr-policies-rag-project)"
    )

    st.divider()

    st.subheader("Conversations")

    # Start a fresh conversation / clear current chat
    if st.button(
        "＋ New conversation",
        use_container_width=True,
        key="new_conversation",
    ):
        start_new_conversation()
        st.rerun()

    st.divider()

    # Retrieve saved conversations
    try:
        conversations = get_recent_conversations(
        user_id=st.session_state.user_id,
        limit=50,
        )
        st.caption(f"{len(conversations)} saved conversations")

    except Exception as exc:
        st.error(
        f"Unable to load conversation history: {type(exc).__name__}"
        )
        conversations = []



    # Conversation entry
    def render_conversation_entry(conversation):

        conversation_id = conversation["id"]
        title = conversation.get(
            "title",
            "Untitled conversation",
        )

        button_title = (
            title if len(title) <= 34
            else title[:31] + "..."
        )

        is_active = (
            st.session_state.session_id == conversation_id
        )

        # Open conversation
        if st.button(
            button_title,
            key=f"open_{conversation_id}",
            help=title,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            load_saved_conversation(conversation_id)

        st.caption(
            f"{conversation.get('message_count', 0)} messages"
        )

        pin_column, delete_column = st.columns(2)

        # Pin / unpin
        with pin_column:
            pin_label = (
                "Unpin"
                if conversation["is_pinned"]
                else "Pin"
            )

            if st.button(
                pin_label,
                key=f"pin_{conversation_id}",
                use_container_width=True,
            ):
                toggle_conversation_pin(conversation)

        # Delete
        with delete_column:
            if st.button(
                "Delete",
                key=f"delete_{conversation_id}",
                use_container_width=True,
            ):
                st.session_state.pending_delete_id = conversation_id
                st.rerun()

        # Delete confirmation
        if (
            st.session_state.pending_delete_id
            == conversation_id
        ):
            st.warning("Permanently delete this conversation?")

            confirm_column, cancel_column = st.columns(2)

            with confirm_column:
                if st.button(
                    "Confirm",
                    key=f"confirm_{conversation_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    remove_saved_conversation(conversation_id)

            with cancel_column:
                if st.button(
                    "Cancel",
                    key=f"cancel_{conversation_id}",
                    use_container_width=True,
                ):
                    st.session_state.pending_delete_id = None
                    st.rerun()

        st.divider()

    # Pinned conversations
    pinned_conversations = [
        item for item in conversations
        if item["is_pinned"]
    ]

    with st.expander(
        f"Pinned conversations ({len(pinned_conversations)})",
        expanded=True,
    ):
        if pinned_conversations:
            for conversation in pinned_conversations:
                render_conversation_entry(conversation)
        else:
            st.caption("No pinned conversations yet.")

    # Unpinned conversation history
    history_conversations = [
        item for item in conversations
        if not item["is_pinned"]
    ]

    with st.expander(
        f"Conversation history ({len(history_conversations)})",
        expanded=True,
    ):
        if history_conversations:
            for conversation in history_conversations:
                render_conversation_entry(conversation)
        else:
            st.caption("No previous conversations yet.")

    st.divider()

    # Sample questions
    st.subheader("Sample questions")

    for index, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(
            sample,
            key=f"sample_question_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = sample
            st.rerun()

    st.divider()

    st.caption("Current conversation ID")
    st.code(
        st.session_state.session_id,
        language=None,
    )


# ==========================================
# MAIN PAGE: CHAT + RIGHT RAGAS PANEL
# ==========================================

chat_column, evaluation_column = st.columns(
    [3.2, 1.6],
    gap="large",
)

if st.session_state.active_view == "manual":
    st.button(
        "Back to HR Policies Assistant",
        icon=":material/arrow_back:",
        on_click=lambda: st.session_state.update(active_view="chat"),
    )


    display_policy_manual()
    st.stop()

    # Render the manual page content here.
    # Move the PDF loading and viewer code from HR_Policy_Manual.py
    # into a reusable function before calling it.
    st.stop()


# ==========================================
# CHAT AREA
# ==========================================

with chat_column:
    # Create this container before the input. New question/answer output
    # is rendered inside it, so the chat input remains underneath.
    messages_area = st.container()

    # Sample questions are transferred into this widget before it is created.
    sample_question = st.session_state.pop("pending_question", None)
    if sample_question:
        st.session_state[CHAT_INPUT_KEY] = sample_question

    typed_question = st.chat_input(
        "Ask a question about your HR policies...",
        key=CHAT_INPUT_KEY,
    )

    if typed_question and typed_question.strip():
        question = typed_question.strip()

        user_message = {
            "role": "user",
            "content": question,
        }
        st.session_state.messages.append(user_message)

        # Render existing history and the just-submitted user question
        # in the container positioned above the chat input.
        with messages_area:
            for message in st.session_state.messages:
                display_message(message)

            with st.chat_message("assistant"):
                with st.spinner(
                    "Retrieving documents and generating your answer..."
                ):
                    try:
                        # Persist the user's message before executing RAG.
                        if API_MODE:
                            if not st.session_state.get("conversation_initialized", False):
                                api_request(
                                    "POST",
                                    "/conversations",
                                    json={"conversation_id": st.session_state.session_id,
                                        },
                                        )
                                st.session_state.conversation_initialized = True
                        else:
                            persist_conversation_message(user_message)

                        data, trace_events = run_chat_question(question)
                        answer = data.get("answer")

                        if not isinstance(answer, str) or not answer.strip():
                            raise ValueError(
                                "The RAG pipeline returned an empty answer."
                            )

                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "sources": data.get("sources", []),
                            "retrieved_contexts": data.get(
                                "retrieved_contexts", []
                            ),
                            "cache": data.get("cache"),
                            "trace": trace_events,
                        }

                    except Exception as exc:
                        import traceback
                        traceback.print_exc()

                        assistant_message = {
                            "role": "assistant",
                            "content": (
                                f"Unable to generate an answer "
                                f"({type(exc).__name__}): {str(exc)}"
                            ),
                            "sources": [],
                            "retrieved_contexts": [],
                            "cache": None,
                            "trace": [],
                        }

                # Display answer and its evidence/trace below the user question.
                st.markdown(assistant_message["content"])
                if assistant_message.get("role") == "assistant":
                    display_assistant_details(assistant_message)

        # Keep answer in session history and Neon for future reruns.
        st.session_state.messages.append(assistant_message)
        try:
            persist_conversation_message(assistant_message)
        except Exception as save_exc:
            st.warning(
                "The answer was generated, but the conversation history "
                f"could not be saved: {type(save_exc).__name__}."
            )
        get_recent_conversations.clear()
        st.rerun()

    else:
        # Normal page render: show saved chat above the chat input.
        with messages_area:
            for message in st.session_state.messages:
                display_message(message)


# ==========================================
# RIGHT SIDEBAR: RAGAS EVALUATION
# ==========================================

with evaluation_column:

    with st.container(
        key="ragas-right-sidebar",
        border=True,
    ):
        display_ragas_dashboard()
