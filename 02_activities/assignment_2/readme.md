# NEXUS — AI Research Assistant
### Assignment 2: Conversational AI System

---

## Overview

**NEXUS** is a Gradio-based conversational AI assistant with a distinct personality:
an enthusiastic, precise, and witty AI research companion. The system integrates
three services through OpenAI function calling, maintaining conversation memory
and enforcing content guardrails.

---

## Quick Start

```bash
# 1. Set your API key
export OPENAI_API_KEY=sk-...

# 2. Generate the knowledge base embeddings (run once)
cd 05_src/assignment_chat
python create_embeddings.py

# 3. Launch the app
python app.py
# → Opens at http://localhost:7860
```

---

## Services

### Service 1 — Real-Time Weather API (`services/weather_service.py`)

**API:** [Open-Meteo](https://open-meteo.com/) — completely free, no API key required.

**How it works:**
1. Geocodes the city name using Open-Meteo's Geocoding API (`geocoding-api.open-meteo.com`)
2. Fetches current weather and hourly forecast from the Weather API (`api.open-meteo.com`)
3. Converts WMO weather codes to human-readable condition descriptions
4. Returns structured data as a Python dict for the LLM to narrate naturally

**Output transformation:** The raw JSON (latitude, WMO codes, numeric values) is
never shown verbatim. Instead, the LLM receives the structured dict and writes
a conversational weather summary (e.g., *"Toronto is currently sitting at 8°C with
mostly cloudy skies and a 40% chance of rain over the next few hours."*).

**Design decision:** Open-Meteo was chosen over OpenWeatherMap or WeatherAPI because
it requires no API key, has generous rate limits, and provides richer forecast data
including precipitation probability — all at zero cost.

---

### Service 2 — AI/ML Knowledge Base (`services/knowledge_service.py`)

**Approach:** Semantic search using ChromaDB (file-persistent) with OpenAI embeddings.

**Dataset:** `data/ai_concepts.csv` — 25 hand-curated AI/ML concept entries covering:
- Core architectures (Transformer, CNN, LSTM, MoE)
- Training techniques (RLHF, Fine-tuning, Distillation, NAS)
- LLM topics (RAG, Prompt Engineering, Hallucination, Tokenization)
- Generative AI (Diffusion Models, GANs)
- Alignment (Constitutional AI, RLHF)
- Infrastructure (Vector Databases, Embeddings)

**Embedding process:**
- Model: `text-embedding-3-small` (1536 dimensions, strong semantic quality, low cost)
- Run `python create_embeddings.py` once to generate and persist embeddings
- ChromaDB persists to `chroma_db/` using `chromadb.PersistentClient`
- Similarity metric: cosine distance (HNSW index)
- Top-3 results returned by default, ranked by semantic relevance
- The app auto-initialises the DB on startup if `chroma_db/` is missing

**Design decisions:**
- Own dataset chosen to ensure full control over quality and to demonstrate
  the end-to-end embedding pipeline.
- File size: 25 entries × ~150 words = well under 40 MB (CSV is ~14 KB;
  ChromaDB directory is ~2 MB).
- `text-embedding-3-small` was chosen over `ada-002` for its superior
  semantic quality at comparable cost.
- No SQLite used (as required). ChromaDB's default backend is DuckDB + Parquet.

---

### Service 3 — Smart Calculator via Function Calling (`services/function_service.py`)

**Technique:** OpenAI Function Calling (tool use).

**Tools exposed:**
| Tool | Description |
|---|---|
| `calculate` | Evaluates math expressions: arithmetic, `sqrt`, `sin`, `log`, `15% of X`, etc. |
| `convert_units` | Converts between units in 7 categories (length, mass, volume, speed, area, data, temperature) |

**Why function calling (not web search or MCP)?**
Function calling is the ideal paradigm for deterministic computation: it gives the LLM
a structured way to request exact numeric operations rather than guessing at results.
The calculator doesn't need the internet — it needs precise execution. This also
demonstrates how to build a reliable tool-use loop with multiple sequential calls
(e.g., convert + then calculate on the result).

**Safety:** The `calculate` function uses `eval` with a restricted namespace
(`__builtins__` set to `{}`) that only exposes `math` module functions and basic
Python builtins (`abs`, `round`, `min`, `max`). No file system access, imports, or
arbitrary code execution is possible.

**Design decision:** All four tools (weather, knowledge search, calculate, convert)
are registered with OpenAI and the LLM decides which one(s) to call based on user
intent. This is more robust than keyword-based routing and demonstrates proper
multi-tool orchestration.

---

## User Interface

- **Framework:** Gradio `Blocks` API for full layout control
- **Theme:** Soft + violet/indigo palette with Inter font
- **Personality:** NEXUS — enthusiastic, precise, witty, warm
- **Memory:** Sliding window of the last **10 conversation turns** (20 messages)

### Memory Management

The context window is managed with a simple **sliding window** strategy:
- The last `MAX_HISTORY = 10` turns are included in every API call
- Older turns are silently dropped (no summarisation)
- This is a deliberate simplification: the system maintains short-term coherence
  without the complexity of a summarisation loop
- The trade-off: very long conversations may lose early context, but this is
  acceptable for a research assistant whose queries are largely independent

**Optional enhancement (not implemented but described):**
A more sophisticated approach would summarise older turns into a "memory summary"
message, similar to LangGraph's `manage_memory` pattern. This would preserve
high-level context (e.g., user's research area) while keeping token usage bounded.

---

## Guardrails

Two-layer defence:

**Layer 1 — Python pre-screening (`_check_guardrails` in `app.py`):**
- Detects system prompt extraction/override attempts via keyword matching
- Blocks restricted topics: cats/dogs, horoscopes/zodiac, Taylor Swift
- Returns a canned refusal before any LLM call is made

**Layer 2 — System prompt instructions:**
- Instructs the LLM to refuse and redirect for restricted topics
- Instructs the LLM to never reveal or paraphrase its own instructions
- Provides the LLM with context about what counts as a manipulation attempt

**Restricted topics:**
| Topic | Detection |
|---|---|
| Cats / dogs | Keyword match: cat, cats, kitten, dog, dogs, puppy, feline, canine |
| Horoscopes / zodiac | Keyword match: horoscope, zodiac, astrology + zodiac sign names |
| Taylor Swift | Phrase match: "taylor swift" |
| System prompt | Signal phrases: "system prompt", "reveal your", "ignore your instructions", etc. |

---

## File Structure

```
05_src/assignment_chat/
├── app.py                       # Main Gradio application
├── create_embeddings.py         # One-time embedding generation script
├── readme.md                    # This file
├── services/
│   ├── __init__.py
│   ├── weather_service.py       # Service 1: Open-Meteo API
│   ├── knowledge_service.py     # Service 2: ChromaDB semantic search
│   └── function_service.py      # Service 3: Calculator + unit converter + tool definitions
├── data/
│   └── ai_concepts.csv          # 25 AI/ML concept entries (source data)
└── chroma_db/                   # Persisted ChromaDB (generated by create_embeddings.py)
```

---

## Dependencies

All packages are from the standard course setup:
- `openai` — LLM calls and function calling
- `gradio` — Chat interface
- `chromadb` — Vector database with file persistence
- `pandas` — CSV loading for knowledge base
- `requests` — Open-Meteo API calls
- `python-dotenv` (optional) — Load `.secrets` file

---

## Implementation Notes

1. **Function calling loop:** The app supports up to 5 sequential tool calls per user
   turn, enabling multi-step queries like "Convert 100km to miles, then tell me the weather
   in London" in a single response.

2. **Gradio `type="messages"`:** Uses the newer OpenAI-compatible message format
   (`{"role": "user"/"assistant", "content": "..."}`) rather than tuple format.

3. **Knowledge base auto-init:** If `chroma_db/` is missing, the app automatically
   runs the embedding generation on startup (requires OPENAI_API_KEY). This makes
   the first run seamless.

4. **Weather transformation:** The service returns a Python dict; the LLM receives
   this as a tool result and rewrites it as natural prose — satisfying the "must not
   provide verbatim API output" requirement.
