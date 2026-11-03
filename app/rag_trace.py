from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_trace_events = ContextVar("rag_trace_events", default=None)


@contextmanager
def capture_rag_trace():
    """Capture execution events for the current RAG request."""
    events = []
    token = _trace_events.set(events)

    try:
        yield events
    finally:
        _trace_events.reset(token)


@contextmanager
def trace_stage(name: str):
    """Record a real pipeline stage and its execution duration."""
    events = _trace_events.get()

    event = {
        "stage": name,
        "status": "running",
        "duration_ms": 0,
    }

    if events is not None:
        events.append(event)

    started = perf_counter()

    try:
        yield event
    except Exception as exc:
        event["status"] = (
            "blocked"
            if type(exc).__name__ == "QuestionRejectedError"
            else "failed"
        )
        event["error_type"] = type(exc).__name__
        raise
    else:
        if event["status"] == "running":
            event["status"] = "completed"
    finally:
        event["duration_ms"] = round(
            (perf_counter() - started) * 1000, 2
        )
