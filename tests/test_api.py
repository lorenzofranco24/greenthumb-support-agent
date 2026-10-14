"""API tests with a scripted fake LLM: no API keys and no model download needed.

Run from the repository root:
    pytest tests -q
"""

import os

os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import agent_core, main


class ScriptedLLM:
    """Fake chat model: decides tool calls from the last user message and tool results."""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        last_human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        question = last_human.content.lower()
        turn = messages[messages.index(last_human) + 1 :]
        tools_done = [m.name for m in turn if isinstance(m, ToolMessage)]

        if "gtpo" in question and "get_order_status" not in tools_done:
            order = next(w for w in last_human.content.replace("?", " ").split() if "GTPO" in w.upper())
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_order_status", "args": {"order_id": order.strip(".,")}, "id": "c1"}],
            )
        if "danneggiat" in question and "escalate_to_human" not in tools_done:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "escalate_to_human",
                        "args": {"reason": "Danno da trasporto", "order_id": "GTPO1046", "priority": "high"},
                        "id": "c2",
                    }
                ],
            )
        if "tulipani" in question and "search_catalog" not in tools_done:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_catalog", "args": {"query": last_human.content}, "id": "c3"}],
            )
        return AIMessage(content="Risposta di test.")


class FakeAssessor:
    """Fake assessor: medium confidence, escalate only when asked about damage."""

    def invoke(self, prompt):
        return agent_core.ResponseAssessment(
            confidence="medium", needs_human="danneggiat" in prompt.lower(), used_sources=[]
        )


@pytest.fixture()
def client(monkeypatch):
    """TestClient with fake embeddings, fake LLM and fake assessor."""
    monkeypatch.setattr(agent_core, "HuggingFaceEmbeddings", lambda **kw: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(agent_core, "ChatLiteLLM", ScriptedLLM)
    monkeypatch.setattr(agent_core, "assessor", FakeAssessor())
    monkeypatch.setattr(agent_core, "SCORE_THRESHOLD", -1.0)  # fake vectors: accept any match
    monkeypatch.setattr(main, "sessions", main.SessionStore())
    agent_core.ESCALATION_LOG.clear()
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_after_startup(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["orders"] == 22 and body["documents"] == 35
    assert body["active_sessions"] == 0


def chat(client, message, session_id=None):
    """POST /chat and return (json body, session id taken from the X-Session-Id header)."""
    response = client.post("/chat", json={"message": message, "session_id": session_id})
    assert response.status_code == 200
    return response.json(), response.headers["X-Session-Id"]


def test_chat_returns_exactly_the_four_spec_fields(client):
    body, session_id = chat(client, "Quando si piantano i tulipani?")
    assert set(body) == {"answer", "confidence", "sources", "needs_human"}
    assert body["answer"] == "Risposta di test."
    assert body["confidence"] in {"low", "medium", "high"}
    assert body["needs_human"] is False
    assert len(session_id) == 12


def test_order_lookup_and_escalation_flag_persists_across_turns(client):
    first, session_id = chat(client, "Il mio ordine GTPO1046 è arrivato con confezioni danneggiate")
    assert first["needs_human"] is True
    assert len(agent_core.ESCALATION_LOG) == 1

    second, same_session_id = chat(client, "Quando mi contattate?", session_id)
    assert same_session_id == session_id
    assert second["needs_human"] is True  # ticket opened in a previous turn
    assert len(agent_core.ESCALATION_LOG) == 1  # no second ticket


def test_sessions_are_isolated(client):
    a, id_a = chat(client, "Il mio ordine GTPO1046 è danneggiato")
    b, id_b = chat(client, "Ciao, come stai?")
    assert id_a != id_b
    assert a["needs_human"] is True
    assert b["needs_human"] is False


def test_memory_tracks_order_ids_and_summarizes_long_conversations(client, monkeypatch):
    monkeypatch.setattr(agent_core, "summarize_messages", lambda msgs, prev="": "Ordini citati: GTPO1042")
    session_id = None
    for text in ["Dov'è il mio ordine GTPO1042?"] + ["Grazie, un'altra domanda"] * 6:
        _, session_id = chat(client, text, session_id)
    state = main.sessions.get_or_create(session_id).state
    assert state.order_ids == ["GTPO1042"]
    assert state.summary == "Ordini citati: GTPO1042"
    assert len(state.messages) <= agent_core.SUMMARY_TRIGGER + 4  # memory is bounded


def test_delete_session_resets_conversation(client):
    _, session_id = chat(client, "Ciao")
    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert client.delete(f"/sessions/{session_id}").status_code == 404
    assert client.get("/health").json()["active_sessions"] == 0


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   "}, {}, {"message": "ok", "session_id": "x" * 100}])
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_long_message_is_truncated(client):
    response = client.post("/chat", json={"message": "a" * 5000})
    assert response.status_code == 200


def test_agent_failure_returns_500_without_leaking_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("secret internal detail")

    monkeypatch.setattr(agent_core, "chat_turn", boom)
    response = client.post("/chat", json={"message": "Ciao"})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_lru_eviction():
    store = main.SessionStore(max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("a")  # a is now the most recently used
    store.get_or_create("c")  # evicts b
    assert len(store) == 2
    assert store.delete("b") is False
    assert store.delete("a") is True
