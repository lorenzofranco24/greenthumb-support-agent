"""GreenThumb support agent: core logic extracted from the project notebook (Steps 1-4).

The module has no Colab dependency. Heavy resources (embedding model, vector store,
LLM client) are created by initialize(), so importing the module is cheap.

Main entry point: chat_turn(state, user_message) -> AgentResponse.
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import litellm
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("greenthumb.agent")
litellm.suppress_debug_info = True

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_URL = (
    "https://raw.githubusercontent.com/lorenzofranco24/"
    "greenthumb-support-agent/refs/heads/main/data/"
)
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_FILES = ["orders.json", "knowledge_base.json"]
# Fixed reference date to make answers and tests reproducible
REFERENCE_DATE = os.getenv("REFERENCE_DATE", "2026-10-04")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "greenthumb_kb"
SCORE_THRESHOLD = 0.55
TOP_K = 3

AGENT_MODEL = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
ASSESSOR_MODEL = os.getenv("ASSESSOR_MODEL", "groq/openai/gpt-oss-20b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "groq/openai/gpt-oss-20b")
MAX_ITERATIONS = 5

SUMMARY_TRIGGER = 10  # above this number of messages the summary is triggered
KEEP_RECENT = 4  # recent messages kept untouched after summarization
TOOL_RESULT_MAX_CHARS = 200

# Resources created by initialize()
orders: list[dict] = []
knowledge_base: list[dict] = []
ORDERS_BY_ID: dict[str, dict] = {}
vectorstore: Chroma | None = None
llm: ChatLiteLLM | None = None
llm_with_tools = None

# Escalation registry (in memory, useful for tests and demos)
ESCALATION_LOG: list[dict] = []


# --------------------------------------------------------------------------- #
# Step 1 - Data
# --------------------------------------------------------------------------- #
def download_data_files(file_names: list[str]) -> None:
    """Download the JSON data files from GitHub and save them in DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        response = requests.get(BASE_URL + file_name, timeout=30)
        response.raise_for_status()
        with open(DATA_DIR / file_name, "w", encoding="utf-8") as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=2)
        logger.info("%s downloaded in %s", file_name, DATA_DIR)


def load_data() -> tuple[list[dict], list[dict]]:
    """Load orders and knowledge base from DATA_DIR, downloading missing files."""
    missing = [name for name in DATA_FILES if not (DATA_DIR / name).exists()]
    if missing:
        download_data_files(missing)
    with open(DATA_DIR / "orders.json", encoding="utf-8") as f:
        orders_data = json.load(f)
    with open(DATA_DIR / "knowledge_base.json", encoding="utf-8") as f:
        kb_data = json.load(f)
    return orders_data, kb_data


# --------------------------------------------------------------------------- #
# Step 2 - Vector store and tools
# --------------------------------------------------------------------------- #
def build_documents(raw_kb: list[dict]) -> list[Document]:
    """Build a list of langchain Document objects from the knowledge base (1 doc = 1 chunk).

    The title is prepended to the text so it contributes to the embedding.
    """
    return [
        Document(
            page_content=f"{item['title']}\n{item['text']}",
            metadata={
                "id": item["id"],
                "category": item["category"],
                "source": item["source"],
                "title": item["title"],
            },
        )
        for item in raw_kb
    ]


@tool
def search_catalog(query: str) -> str:
    """Search the company knowledge base (product sheets, growing guides, returns,
    shipping and warranty policies) for information relevant to the question.

    Use this tool for any question about products, prices, planting and plant care,
    shipping times and costs, returns, warranties or escalation rules.
    Do NOT use it for the status of a specific order.

    Args:
        query: the customer's question in natural language, in Italian.

    Returns:
        The most relevant documents, each with its source file, or a message
        stating that no relevant document was found.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=TOP_K)
    relevant = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]

    if not relevant:
        return "Nessun documento rilevante trovato nella knowledge base."

    blocks = [
        f"[source: {doc.metadata['source']} | score: {score:.2f}]\n{doc.page_content}"
        for doc, score in relevant
    ]
    return "\n\n".join(blocks)


@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a specific customer order by its order ID.

    Use this tool when the customer asks where an order is, when it will arrive,
    or about a delay, a cancellation, a return or a damaged delivery.
    Do NOT use it for questions about products, plant care or company policies.

    Args:
        order_id: the order ID only, for example GTPO1042.

    Returns:
        Status, dates, tracking code, items, total and any notes for the order,
        or a message stating that the order was not found.
    """
    order_id = order_id.strip().upper()
    order = ORDERS_BY_ID.get(order_id)
    if order is None:
        return f"Ordine {order_id} non trovato."

    items = ", ".join(f"{i['qty']}x {i['name']}" for i in order["items"])
    lines = [
        f"Ordine {order['order_id']}",
        f"Stato: {order['status']}",
        f"Data ordine: {order['order_date']}",
        f"Consegna stimata: {order['estimated_delivery'] or 'Non disponibile'}",
        f"Consegnato il: {order['delivered_on'] or 'Non ancora consegnato'}",
        f"Tracking: {order['tracking_code'] or 'Non ancora disponibile'}",
        f"Prodotti: {items}",
        f"Totale: {order['total_eur']:.2f} €",
    ]
    if order["notes"]:
        lines.append(f"Note: {order['notes']}")
    return "\n".join(lines)


def next_working_day(from_date: str) -> str:
    """Return the next Monday-Friday date after from_date (ISO format)."""
    day = date.fromisoformat(from_date) + timedelta(days=1)
    while day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        day += timedelta(days=1)
    return day.isoformat()


@tool
def escalate_to_human(
    reason: str,
    order_id: str | None = None,
    priority: Literal["normal", "high"] = "normal",
) -> str:
    """Hand the conversation over to a human customer support operator.

    Use this tool ONLY when a first-level assistant cannot or must not resolve the case:
    - damage during transport or defective goods reported by the customer
    - refund requests outside the returns policy
    - an order that cannot be found after checking the order ID
    - a customer threatening legal action
    - an order delayed by more than 3 days
    - a disputed amount above 100 EUR
    Do NOT use it for informational questions that the knowledge base can answer.

    Args:
        reason: one short sentence explaining why a human is needed.
        order_id: numeric order ID if the case concerns an order, otherwise omit it.
        priority: "high" for transport damage, legal threats or amounts above
            100 EUR, otherwise "normal".

    Returns:
        A confirmation with the ticket ID and the expected handling time.
    """
    ticket_id = uuid.uuid4().hex[:8]
    ESCALATION_LOG.append(
        {
            "ticket_id": ticket_id,
            "reason": reason,
            "order_id": order_id,
            "priority": priority,
            "created_at": REFERENCE_DATE,
        }
    )
    return (
        f"Segnalazione creata. Ticket: {ticket_id} (priorità {priority}). "
        f"Gli operatori sono disponibili dal lunedì al venerdì, 9:00-18:00. "
        f"Fuori orario la presa in carico avviene entro le 10:00 del "
        f"{next_working_day(REFERENCE_DATE)}"
    )


TOOLS = [search_catalog, get_order_status, escalate_to_human]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# --------------------------------------------------------------------------- #
# Step 3 - Memory
# --------------------------------------------------------------------------- #
ORDER_ID_PATTERN = re.compile(r"GTPO\d{4}\b", re.IGNORECASE)

SUMMARY_PROMPT = """Sei un assistente che mantiene il riassunto di una conversazione di supporto clienti.
Aggiorna il RIASSUNTO PRECEDENTE integrando i NUOVI MESSAGGI. Non riscrivere da zero:
i campi "Ordini citati" e "Azioni già eseguite" sono cumulativi.

Regole:
- Riporta LETTERALMENTE ID d'ordine, date, importi e numeri di ticket.
- Nel campo "Ordini citati" usa esattamente questo elenco: {order_ids}
- Se un campo non ha informazioni scrivi "nessuno". Non inventare.
- Massimo 120 parole, in italiano.
- Rispondi SOLO con lo schema, senza premesse.

Schema:
Ordini citati: ...
Prodotti/argomenti: ...
Problema del cliente: ...
Azioni già eseguite: ...
Domanda ancora aperta: ...

RIASSUNTO PRECEDENTE:
{previous_summary}

NUOVI MESSAGGI:
{transcript}"""

RETRY_WAIT_PATTERN = re.compile(r"try again in ([\d.]+)s")


def extract_order_ids(text: str) -> list[str]:
    """Extract order IDs (format GTPO + 4 digits) from a text.

    IDs are normalized to uppercase and returned without duplicates,
    in order of first appearance.
    """
    matches = ORDER_ID_PATTERN.findall(text)
    return list(dict.fromkeys(m.upper() for m in matches))


def content_to_text(content) -> str:
    """Convert a message content (string or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        ]
        return " ".join(p for p in parts if p)
    return str(content)


def format_transcript(messages: list[BaseMessage]) -> str:
    """Turn a list of LangChain messages into a readable plain-text transcript.

    Knowledge-base search results are truncated; all other tool results
    (order status, escalation tickets) are kept in full.
    """
    lines = []
    for msg in messages:
        text = content_to_text(msg.content).strip()
        if isinstance(msg, HumanMessage):
            lines.append(f"Cliente: {text}")
        elif isinstance(msg, AIMessage):
            if text:
                lines.append(f"Assistente: {text}")
            for call in msg.tool_calls:
                args = json.dumps(call["args"], ensure_ascii=False)
                lines.append(f"Assistente chiama {call['name']} con {args}")
        elif isinstance(msg, ToolMessage):
            if msg.name == "search_catalog" and len(text) > TOOL_RESULT_MAX_CHARS:
                text = text[:TOOL_RESULT_MAX_CHARS] + "..."
            lines.append(f"Risultato {msg.name}: {text}")
    return "\n".join(lines)


def call_with_backoff(fn, max_attempts: int = 4):
    """Call fn(); on a rate-limit error wait the time suggested by the provider and retry."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except litellm.RateLimitError as exc:
            if attempt == max_attempts:
                raise
            match = RETRY_WAIT_PATTERN.search(str(exc))
            wait = float(match.group(1)) + 1 if match else 15 * attempt
            logger.warning(
                "Rate limit: waiting %.0fs (attempt %d/%d)", wait, attempt, max_attempts
            )
            time.sleep(wait)


def summarize_messages(messages: list[BaseMessage], previous_summary: str = "") -> str:
    """Update a structured summary with the messages leaving the recent window.

    Order IDs come from a regex, not from the model, and are passed in the
    prompt so that the model copies them literally.
    """
    transcript = format_transcript(messages)
    previous = previous_summary or "Nessun riassunto precedente."
    order_ids = extract_order_ids(f"{previous_summary}\n{transcript}")

    prompt = SUMMARY_PROMPT.format(
        order_ids=", ".join(order_ids) if order_ids else "nessuno",
        previous_summary=previous,
        transcript=transcript,
    )
    response = call_with_backoff(
        lambda: litellm.completion(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            reasoning_effort="low",
            max_tokens=1000,
        )
    )
    return response["choices"][0].message.content.strip()


def find_cut_index(messages: list[BaseMessage], keep_recent: int = KEEP_RECENT) -> int | None:
    """Return the index where the recent window should start.

    The window always starts on a HumanMessage, so a tool call is never separated
    from its result. The cut moves backwards from len(messages) - keep_recent,
    which keeps the window at keep_recent messages or slightly more.
    Returns None if there is nothing useful to cut.
    """
    target = len(messages) - keep_recent
    for i in range(target, 0, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return None


def update_order_ids(messages: list[BaseMessage], order_ids: list[str]) -> list[str]:
    """Add the order IDs mentioned by the customer to the authoritative list."""
    found = list(order_ids)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            found.extend(extract_order_ids(content_to_text(msg.content)))
    return list(dict.fromkeys(found))


def manage_memory(
    messages: list[BaseMessage],
    summary: str,
    order_ids: list[str],
) -> tuple[list[BaseMessage], str, list[str]]:
    """Apply the summary-buffer strategy to the short-term memory.

    Order IDs are updated at every turn. When the history exceeds SUMMARY_TRIGGER
    messages, older messages are summarized and only the recent window is kept.
    If the summarization call fails, it falls back to plain trimming: the old
    messages are dropped but the order IDs are preserved.
    """
    order_ids = update_order_ids(messages, order_ids)

    if len(messages) <= SUMMARY_TRIGGER:
        return messages, summary, order_ids

    cut = find_cut_index(messages)
    if cut is None:
        return messages, summary, order_ids

    old_messages, recent_messages = messages[:cut], messages[cut:]
    try:
        summary = summarize_messages(old_messages, summary)
    except Exception as exc:  # fallback: plain trimming, the previous summary is kept
        logger.warning("Summarization failed (%s): plain trimming", type(exc).__name__)
    return recent_messages, summary, order_ids


# --------------------------------------------------------------------------- #
# Step 4 - ReAct agent and structured output
# --------------------------------------------------------------------------- #
class AgentResponse(BaseModel):
    """Structured final response returned by the support agent."""

    answer: str = Field(description="Risposta al cliente, in italiano, chiara e concisa.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Affidabilità della risposta: high, medium o low."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="File della knowledge base usati per rispondere.",
    )
    needs_human: bool = Field(description="True se il caso richiede un operatore umano.")


class ResponseAssessment(BaseModel):
    """Model-generated assessment of the final answer (not the answer itself)."""

    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "'high' se la risposta è interamente supportata dai risultati dei tool; "
            "'medium' se solo in parte; 'low' se le informazioni mancano, non sono "
            "state trovate o la risposta è incerta."
        )
    )
    needs_human: bool = Field(
        description="True se il cliente deve essere preso in carico da un operatore umano."
    )
    used_sources: list[str] = Field(
        default_factory=list,
        description=(
            "File 'source' (come compaiono nei risultati di search_catalog) "
            "effettivamente usati per scrivere la risposta. Vuoto se nessuno."
        ),
    )


SYSTEM_PROMPT_TEMPLATE = """Sei un assistente di supporto clienti di primo livello per un negozio online di giardinaggio.
La data di oggi è {today}: usala per interpretare "oggi", "domani" e le scadenze.

Strumenti:
- search_catalog: prodotti, prezzi, coltivazione, spedizioni, resi, garanzie, regole di escalation.
- get_order_status: stato di un ordine. Passa l'ID ESATTAMENTE come lo scrive il cliente (formato GTPO seguito da 4 cifre). Se il cliente parla di un ordine senza darne l'ID: se in "Ordini citati finora" c'è un solo ordine usa quello; se ce ne sono più di uno o nessuno, chiedi di quale si tratta.
- escalate_to_human: passaggio a un operatore umano.

Regole:
1. Se la domanda ha più parti, usa i tool necessari per OGNI parte prima di rispondere.
2. Rispondi solo con informazioni ottenute dai tool: riporta fatti, valori e passaggi presenti nei risultati, senza aggiungere consigli, dosi, procedure o dettagli tuoi che non compaiono nelle fonti, anche se ti sembrano di buon senso. Se la knowledge base non contiene la risposta, dillo chiaramente senza inventare.
3. Se un risultato di ricerca non è pertinente alla domanda, ignoralo.
4. Escala a un operatore nei casi previsti dalla policy: danni da trasporto, rimborsi fuori policy, ordine non trovato, minacce legali, ordini in ritardo di oltre 3 giorni, importi contestati oltre 100 EUR. Il ritardo si calcola come differenza tra la data di consegna originaria (indicata nelle note dell'ordine) e la nuova data stimata. Una richiesta di rimborso o di indennizzo per un ritardo o per un disservizio è sempre fuori policy, perché la policy dei resi riguarda solo i prodotti non utilizzati restituiti: se il cliente cita un ordine, verifica prima il suo stato con get_order_status, poi escala senza cercare altre regole. Non escalare per domande informative.
5. Non aprire una seconda escalation per un caso già segnalato (controlla "Azioni già eseguite" e le note dell'ordine). Se la pratica è già aperta, informa il cliente che un operatore se ne sta occupando: il caso resta comunque da gestire da un operatore.
6. Rispondi in italiano, in modo cordiale e conciso.
7. Quando escali un caso, o informi che una pratica è già aperta, cita sempre l'ID dell'ordine e il motivo (per esempio il danno da trasporto, oppure il ritardo con la data originaria e la nuova data stimata). Se esiste già una pratica, di' che è già in carico a un operatore invece di dire che l'hai appena segnalata.

Ordini citati finora: {order_ids}

Riassunto della conversazione precedente:
{summary}"""

ASSESSMENT_PROMPT = """Valuta la risposta di un assistente di supporto clienti.

DOMANDA DEL CLIENTE:
{question}

RISULTATI DEI TOOL IN QUESTO TURNO:
{observations}

RISPOSTA DELL'ASSISTENTE:
{answer}

Compila la valutazione. Per used_sources indica solo i file che hanno davvero
contribuito alla risposta, ignorando risultati non pertinenti.

Rispondi SOLO con un oggetto JSON con esattamente questi campi:
{{"confidence": "low" | "medium" | "high", "needs_human": true | false, "used_sources": ["file1", "file2"]}}"""

FORCED_ANSWER_INSTRUCTION = (
    "Non puoi più usare strumenti. Rispondi ora al cliente usando SOLO le "
    "informazioni già raccolte qui sopra; se qualcosa manca, dichiaralo."
)
FALLBACK_ANSWER = (
    "Mi scuso, non riesco a completare la risposta. Ti consiglio di contattare "
    "un operatore del supporto."
)

SOURCE_PATTERN = re.compile(r"\[source:\s*([^\s|\]]+)")
PENDING_MARKER = "in attesa di operatore"


def build_system_prompt(summary: str, order_ids: list[str]) -> str:
    """Build the system prompt with the reference date, cited orders and summary."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=REFERENCE_DATE,
        order_ids=", ".join(order_ids) if order_ids else "nessuno",
        summary=summary or "nessuno (inizio conversazione)",
    )


def execute_tool_call(call: dict) -> ToolMessage:
    """Run one tool call requested by the model and wrap the result in a ToolMessage.

    Errors (unknown tool, invalid arguments, exceptions) are returned as text,
    so the model can see them and correct itself instead of crashing the loop.
    """
    tool_fn = TOOLS_BY_NAME.get(call["name"])
    if tool_fn is None:
        content = f"Errore: il tool '{call['name']}' non esiste."
    else:
        try:
            content = tool_fn.invoke(call["args"])
        except Exception as exc:
            content = f"Errore nell'esecuzione del tool: {str(exc)[:300]}"
    return ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])


def force_final_answer(
    history: list[BaseMessage],
    new_messages: list[BaseMessage],
    system_prompt: str,
) -> AIMessage:
    """Force a final answer without tools, once the iteration limit is reached.

    The history is flattened into plain text, so the model has no tool-call
    pattern to continue. If the call fails, a safe fallback message is returned.
    """
    observations = "\n\n".join(
        f"Risultato {m.name}:\n{content_to_text(m.content)}"
        for m in new_messages
        if isinstance(m, ToolMessage)
    )
    prompt = (
        f"CONVERSAZIONE:\n{format_transcript(history)}\n\n"
        f"INFORMAZIONI RACCOLTE IN QUESTO TURNO:\n{observations or 'nessuna'}\n\n"
        f"{FORCED_ANSWER_INSTRUCTION}"
    )
    try:
        return call_with_backoff(
            lambda: llm.invoke([SystemMessage(system_prompt), HumanMessage(prompt)])
        )
    except Exception as exc:
        logger.warning("Forced answer failed (%s): using fallback", type(exc).__name__)
        return AIMessage(content=FALLBACK_ANSWER)


def run_react_loop(
    history: list[BaseMessage],
    system_prompt: str,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = False,
) -> list[BaseMessage]:
    """Run the ReAct loop (reason -> act -> observe) and return the new messages.

    The loop ends when the model answers without tool calls. If max_iterations
    is reached, a final call without tools forces the model to answer.
    Rate-limit errors are retried with the wait suggested by the provider.
    The input history is not modified: the caller decides what to keep.
    """
    new_messages: list[BaseMessage] = []

    for step in range(1, max_iterations + 1):
        ai_message = call_with_backoff(
            lambda: llm_with_tools.invoke(
                [SystemMessage(system_prompt)] + history + new_messages
            )
        )
        new_messages.append(ai_message)

        if not ai_message.tool_calls:
            if verbose:
                print(f"  [passo {step}] risposta finale")
            return new_messages

        for call in ai_message.tool_calls:
            result = execute_tool_call(call)
            new_messages.append(result)
            if verbose:
                args = json.dumps(call["args"], ensure_ascii=False)
                preview = str(result.content).replace("\n", " ")[:80]
                print(f"  [passo {step}] {call['name']}({args}) -> {preview}...")

    if verbose:
        print(f"  ⚠️ limite di {max_iterations} passaggi raggiunto: risposta forzata")
    new_messages.append(force_final_answer(history, new_messages, system_prompt))
    return new_messages


class JsonAssessor:
    """Fills ResponseAssessment using JSON mode, validated with Pydantic.

    Uses a separate model from the agent, so it does not consume the agent's
    token quota. Exposes .invoke(prompt) like a LangChain runnable, so
    build_agent_response does not need to change.
    """

    MAX_ATTEMPTS = 3

    def invoke(self, prompt: str) -> ResponseAssessment:
        """Call the model in JSON mode and validate the result against the schema.

        Retries when the model returns invalid JSON (a transient failure of
        reasoning models); the fallback in build_agent_response is used only
        after MAX_ATTEMPTS failures.
        """
        last_error = None
        for _ in range(self.MAX_ATTEMPTS):
            try:
                response = call_with_backoff(
                    lambda: litellm.completion(
                        model=ASSESSOR_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=2000,  # reasoning models spend tokens while thinking
                        response_format={"type": "json_object"},
                    )
                )
                return ResponseAssessment.model_validate_json(
                    response.choices[0].message.content
                )
            except (litellm.BadRequestError, ValidationError) as exc:
                last_error = exc
        raise last_error


assessor = JsonAssessor()


def extract_sources(new_messages: list[BaseMessage]) -> list[str]:
    """Return the knowledge-base files returned by search_catalog in this turn."""
    found = []
    for msg in new_messages:
        if isinstance(msg, ToolMessage) and msg.name == "search_catalog":
            found.extend(SOURCE_PATTERN.findall(content_to_text(msg.content)))
    return list(dict.fromkeys(found))


def escalation_ticket_created(new_messages: list[BaseMessage]) -> bool:
    """True if escalate_to_human actually created a ticket in the given messages."""
    return any(
        isinstance(msg, ToolMessage)
        and msg.name == "escalate_to_human"
        and content_to_text(msg.content).startswith("Segnalazione creata")
        for msg in new_messages
    )


def case_pending_with_operator(new_messages: list[BaseMessage]) -> bool:
    """True if an order looked up in the given messages already has a case waiting for an operator."""
    return any(
        isinstance(msg, ToolMessage)
        and msg.name == "get_order_status"
        and PENDING_MARKER in content_to_text(msg.content).lower()
        for msg in new_messages
    )


def build_agent_response(
    history: list[BaseMessage], new_messages: list[BaseMessage]
) -> AgentResponse:
    """Build the structured response, enforcing sources and needs_human in code.

    - answer: the final text of the ReAct loop, not rewritten.
    - sources: only files actually returned by search_catalog and used by the model.
    - needs_human: forced to True if a ticket was created in this conversation or if the
      order already has a case waiting for an operator.
    - if the assessment call fails: confidence 'low' and needs_human True.
    """
    answer = content_to_text(new_messages[-1].content).strip()
    question = next(
        (content_to_text(m.content) for m in reversed(history) if isinstance(m, HumanMessage)),
        "",
    )

    retrieved = extract_sources(new_messages)

    current_observations = [
        f"- {m.name}: {content_to_text(m.content)[:400]}"
        for m in new_messages
        if isinstance(m, ToolMessage)
    ]
    if retrieved:
        current_observations.append(f"- Fonti recuperate in questo turno: {', '.join(retrieved)}")

    past_observations = [
        f"- (turno precedente) {m.name}: {content_to_text(m.content)[:400]}"
        for m in history
        if isinstance(m, ToolMessage)
    ]
    observations = "\n".join(past_observations + current_observations) or "nessun tool usato"

    forced_human = (
        escalation_ticket_created(new_messages)
        or escalation_ticket_created(history)  # ticket already opened in a previous turn
        or case_pending_with_operator(new_messages)
        or case_pending_with_operator(history)  # order already in a case in a previous turn
    )

    try:
        assessment = assessor.invoke(
            ASSESSMENT_PROMPT.format(question=question, observations=observations, answer=answer)
        )
        confidence, needs_human = assessment.confidence, assessment.needs_human
        sources = [s for s in assessment.used_sources if s in retrieved]
    except Exception as exc:
        logger.warning("Assessment failed (%s): %s", type(exc).__name__, str(exc)[:200])
        confidence, needs_human, sources = "low", True, []

    return AgentResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        needs_human=needs_human or forced_human,
    )


@dataclass
class ConversationState:
    """Short-term memory of one conversation (one instance per session)."""

    messages: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    order_ids: list[str] = field(default_factory=list)


def chat_turn(state: ConversationState, user_message: str, verbose: bool = False) -> AgentResponse:
    """Run one conversation turn: memory management, ReAct loop and structured response.

    The state is updated in place: the new messages (including tool calls and
    results) are appended to the history after the response is built.
    """
    if llm_with_tools is None or vectorstore is None:
        raise RuntimeError("Agent not initialized: call initialize() first.")

    state.messages.append(HumanMessage(user_message))
    state.messages, state.summary, state.order_ids = manage_memory(
        state.messages, state.summary, state.order_ids
    )
    system_prompt = build_system_prompt(state.summary, state.order_ids)
    new_messages = run_react_loop(state.messages, system_prompt, verbose=verbose)
    response = build_agent_response(state.messages, new_messages)
    state.messages.extend(new_messages)
    return response


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
def initialize(embeddings=None) -> dict:
    """Load data, build the vector store and create the LLM client.

    Call it once at application startup. The `embeddings` argument lets tests
    inject a lightweight embedding model instead of downloading BAAI/bge-m3.

    Returns:
        Basic counters (orders, knowledge base documents) for the health check.
    """
    global orders, knowledge_base, ORDERS_BY_ID, vectorstore, llm, llm_with_tools

    if "GROQ_API_KEY" not in os.environ and AGENT_MODEL.startswith("groq/"):
        raise RuntimeError("GROQ_API_KEY is not set: add it to the .env file.")

    orders, knowledge_base = load_data()
    ORDERS_BY_ID = {o["order_id"]: o for o in orders}

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            # Unit-length vectors: dot product equals cosine similarity
            encode_kwargs={"normalize_embeddings": True},
        )

    if vectorstore is not None:
        vectorstore.delete_collection()
    vectorstore = Chroma.from_documents(
        documents=build_documents(knowledge_base),
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    llm = ChatLiteLLM(model=AGENT_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    logger.info(
        "Agent ready: %d orders, %d KB documents", len(orders), vectorstore._collection.count()
    )
    return {"orders": len(orders), "documents": vectorstore._collection.count()}
