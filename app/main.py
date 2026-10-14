"""FastAPI service exposing the GreenThumb customer support agent.

Run from the repository root:
    uvicorn app.main:app --reload

Interactive documentation is available at http://127.0.0.1:8000/docs
"""

import logging
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator

load_dotenv()  # reads GROQ_API_KEY from the .env file before the agent is created

from app import agent_core  # noqa: E402  (must be imported after load_dotenv)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("greenthumb.api")

MAX_SESSIONS = 200  # when exceeded, the least recently used session is dropped
MAX_MESSAGE_CHARS = 1000


class Session:
    """One conversation: its memory and a lock that serializes its turns."""

    def __init__(self) -> None:
        self.state = agent_core.ConversationState()
        self.lock = threading.Lock()


class SessionStore:
    """In-memory store of conversations, keyed by session_id (LRU eviction)."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        """Return the session for session_id, creating it if it does not exist."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session()
                self._sessions[session_id] = session
                while len(self._sessions) > self._max_sessions:
                    self._sessions.popitem(last=False)
            self._sessions.move_to_end(session_id)
            return session

    def delete(self, session_id: str) -> bool:
        """Remove a session. Returns False if it did not exist."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)


sessions = SessionStore()
startup_info: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build the vector store once, when the server starts."""
    logger.info("Starting up: loading embedding model and knowledge base...")
    startup_info.update(agent_core.initialize())
    yield


app = FastAPI(
    title="GreenThumb Support Agent",
    description="ReAct customer support agent (RAG + order lookup + human escalation).",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(description="Customer message, in Italian.")
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Conversation ID (from the X-Session-Id header of a previous response). Omit it to start a new conversation.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Strip whitespace, reject empty messages and truncate very long ones."""
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value[:MAX_MESSAGE_CHARS]


@app.post("/chat", response_model=agent_core.AgentResponse)
def chat(request: ChatRequest, response: Response) -> agent_core.AgentResponse:
    """Run one conversation turn and return the structured response.

    The body contains exactly the four fields answer, confidence, sources and
    needs_human. The conversation ID travels in the X-Session-Id response header:
    send it back in the "session_id" field of the next request to continue the
    same conversation.

    The endpoint is a plain `def`, so FastAPI runs it in a thread pool and the
    blocking LLM calls do not freeze the server. Turns of the same session are
    serialized with a lock.
    """
    session_id = request.session_id or uuid.uuid4().hex[:12]
    session = sessions.get_or_create(session_id)

    with session.lock:
        try:
            result = agent_core.chat_turn(session.state, request.message)
        except litellm.RateLimitError:
            logger.warning("Rate limit reached for session %s", session_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The language model is rate limited. Retry in a few seconds.",
            )
        except Exception:
            logger.exception("Agent error in session %s", session_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error while processing the message.",
            )

    response.headers["X-Session-Id"] = session_id
    return result


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> Response:
    """Reset a conversation by deleting its memory."""
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health")
def health() -> dict:
    """Liveness check with a few counters."""
    return {
        "status": "ok",
        "orders": startup_info.get("orders"),
        "documents": startup_info.get("documents"),
        "active_sessions": len(sessions),
    }
