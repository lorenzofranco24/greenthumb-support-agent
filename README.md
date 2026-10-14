# GreenThumb – agente di supporto clienti (FastAPI)

API locale che espone l'agente ReAct del notebook (RAG sul catalogo, stato ordini,
escalation a operatore, memoria di conversazione) con output JSON strutturato.

## Struttura

```
app/agent_core.py   logica dell'agente (Step 1-4 del notebook, senza dipendenze Colab)
app/main.py         API FastAPI: sessioni, validazione, gestione errori
data/               orders.json, knowledge_base.json (scaricati da GitHub se mancanti)
tests/test_api.py   test dell'API con un LLM finto (nessuna chiave necessaria)
```

## Avvio

Requisiti: Python 3.10+, una chiave API Groq.

```bash
git clone https://github.com/lorenzofranco24/greenthumb-support-agent.git
cd greenthumb-support-agent

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# apri .env e inserisci GROQ_API_KEY

uvicorn app.main:app --reload
```

Al primo avvio il server scarica il modello di embedding `BAAI/bge-m3` (circa 2 GB) e costruisce
il vector store: l'avvio può richiedere alcuni minuti. Il server è pronto quando compare
`Application startup complete`. Documentazione interattiva: http://127.0.0.1:8000/docs

## Uso

`POST /chat` riceve `message` e, per proseguire una conversazione, `session_id`. Il corpo della risposta
contiene esattamente i quattro campi dell'output strutturato; l'ID della conversazione viaggia
nell'header `X-Session-Id` (se non invii `session_id`, ne viene creato uno nuovo).

```bash
# Primo turno: -i mostra anche gli header, tra cui X-Session-Id
curl -i -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Il mio ordine GTPO1046 è arrivato con due confezioni danneggiate"}'

# Secondo turno: stessa conversazione (sostituisci SESSION_ID con il valore dell'header)
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "Quando mi contattate?"}'
```

Risposta (corpo):

```json
{
  "answer": "...",
  "confidence": "high",
  "sources": ["policies/resi.md"],
  "needs_human": true
}
```

| Endpoint | Descrizione |
|---|---|
| `POST /chat` | un turno di conversazione, restituisce il JSON strutturato |
| `DELETE /sessions/{session_id}` | azzera la memoria di una conversazione |
| `GET /health` | stato del servizio e contatori |

Errori: `422` richiesta non valida (messaggio vuoto, `session_id` troppo lungo),
`503` limite di richieste del modello (riprovare tra qualche secondo), `500` errore interno.

## Test

```bash
pytest tests -q
```

I test usano un LLM e degli embedding finti: verificano API, sessioni, memoria ed escalation
senza chiavi né download.

## Scelte e limiti

- Le sessioni sono in memoria (massimo 200, poi viene scartata la meno recente): al riavvio si
  perdono. Per una produzione servirebbe uno store esterno (es. Redis).
- Nessuna autenticazione: il tool ordini non verifica l'identità del cliente. Solo per uso locale/demo.
- Gli endpoint sono sincroni e girano nel thread pool di FastAPI; i turni della stessa sessione sono
  serializzati con un lock.
- La data di riferimento è fissa (`REFERENCE_DATE=2026-10-04`) per rendere riproducibili le risposte.
- Il piano gratuito di Groq ha limiti di token al minuto e al giorno: sotto carico può rispondere `503`.
