# Social Robot Semantic Memory Generator

A multimodal pipeline that turns social-robot interaction episodes into **semantic memories** — structured, queryable representations of meaning, not just raw data — using [Google ADK](https://google.github.io/adk-docs/) and Gemini.

## Origin

This repository is adapted from the [Always-On Memory Agent](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/agents/always-on-memory-agent) reference implementation in `GoogleCloudPlatform/generative-ai`. That design (file watcher → LLM ingestion → SQLite store → periodic consolidation) is repurposed here specifically for social robot semantic memory generation, with targeted changes to ingestion prompts, file routing, importance scoring, consolidation output, and an added semantic fact extraction layer.

This project is not affiliated with or endorsed by Google.

## Why This Exists

Social robots accumulate rich interaction data — structured episode logs, conversation transcripts, audio recordings, and video footage — but have no standard way to turn that raw data into durable **semantic memory**. Semantic memory captures the *meaning* of interactions: who was involved, what was said or expressed, what preferences or emotions were present, and what patterns emerge across episodes. This pipeline ingests episode files and uses Gemini to generate exactly that — structured semantic memories and atemporal facts stored in SQLite and queryable at any time.

## How It Works

```
inbox/
  episode_001.json   ──► [ingest_agent]        ──► memories table (SQLite)
  session_02.txt     ──►  Gemini LLM                │
  audio_03.mp3       ──►  multimodal                ▼
  video_04.mp4       ──►            [fact_extractor_agent] ──► facts table
                                                    │         (entity · attribute · value
                                                    │          confidence · evidence_count)
                                    [consolidate_agent]  (every 30 min)
                                                    │
                                                    ▼
                                            consolidations table
                                            (cross-episode insights)
                                                    │
                                                    ▼
                                            HTTP API  :8888
```

1. **Watch** — polls `./inbox/` every 5 seconds for new files.
2. **Ingest** — each file is routed by type and sent to the `ingest_agent` with a social-episode-aware prompt. Gemini extracts a structured episodic memory and writes it to the `memories` table.
3. **Extract facts** — immediately after ingest, the `fact_extractor_agent` reads the episode description and extracts atemporal semantic facts (preferences, habits, communication style, role) into the `facts` table. If the same fact is observed again in a later episode, its confidence score increases rather than creating a duplicate row.
4. **Consolidate** — every 30 minutes the `consolidate_agent` reads unconsolidated memories, finds cross-episode patterns, and writes a higher-level insight record.
5. **Query** — the HTTP API lets you query across facts, episodic memories, and consolidation insights at any time using natural language.

## Agent Architecture

The system uses five Google ADK agents under a single orchestrator:

| Agent | Role |
|---|---|
| `memory_orchestrator` | Routes all requests; always chains ingest → fact extraction for new episodes |
| `ingest_agent` | Processes episode files into structured episodic memories with summary, entities, topics, and importance score |
| `fact_extractor_agent` | Extracts atemporal semantic facts from each episode and upserts them into the `facts` table |
| `consolidate_agent` | Synthesizes unconsolidated memories into cross-episode insights (runs on timer or on demand) |
| `query_agent` | Answers natural language questions by drawing from facts, memories, and consolidation history |

## Supported Input Types

| Type | Extensions | Handling |
|---|---|---|
| **Episode data** | `.json` | Parsed and rendered as structured interaction prose; LLM extracts social meaning from fields |
| **Transcripts** | `.txt` | Treated as human-robot conversation transcripts; LLM focuses on names, preferences, notable statements |
| **Other text** | `.md`, `.csv`, `.log`, `.xml`, `.yaml`, `.yml` | Standard text ingestion with light framing |
| **Audio** | `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac` | Sent multimodal; LLM extracts voices, tone, spoken content, social dynamics |
| **Video** | `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv` | Sent multimodal; LLM extracts activity, participants, key moments |
| **Image** | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.svg` | Sent multimodal; LLM describes scene and participants |
| **Documents** | `.pdf` | Sent multimodal; LLM extracts content |

Files larger than 20 MB are skipped (Gemini inline limit). Text files larger than 12,000 characters are truncated; a warning is logged when this occurs.

## Memory Structure

### Episodic memories (`memories` table)

Each ingested episode produces one row:

| Field | Type | Description |
|---|---|---|
| `id` | integer | Auto-increment primary key |
| `source` | text | Filename or `"api"` |
| `raw_text` | text | Full LLM description of the episode |
| `summary` | text | 1–2 sentence social summary |
| `entities` | JSON array | People, locations, objects central to the interaction |
| `topics` | JSON array | 2–4 interaction-type tags (e.g. `greeting`, `preference-expressed`) |
| `connections` | JSON array | Links to related memories added during consolidation |
| `importance` | float 0–1 | Social significance score |
| `created_at` | ISO 8601 | Ingestion timestamp (UTC) |
| `consolidated` | boolean | Whether this memory has been through consolidation |

**Importance scale:**

| Range | Meaning |
|---|---|
| 0.8 – 1.0 | Strong emotional content, first meeting, explicit preference/dislike, conflict, personal disclosure |
| 0.5 – 0.7 | Routine but meaningful interaction, clear task completion, sustained conversation |
| 0.2 – 0.4 | Brief or uneventful exchange with little memorable content |

### Semantic facts (`facts` table)

Each atemporal fact about a person, place, or thing:

| Field | Type | Description |
|---|---|---|
| `id` | integer | Auto-increment primary key |
| `entity` | text | The named person, place, or thing (e.g. `"Alex Chen"`) |
| `attribute` | text | Snake_case aspect (e.g. `preferred_speech_speed`, `coffee_preference`) |
| `value` | text | Observed value (e.g. `"slow"`, `"oat milk"`) |
| `confidence` | float 0–1 | Grows toward 1.0 with each confirming observation |
| `evidence_count` | integer | Number of episodes that support this fact |
| `first_seen` | ISO 8601 | When this fact was first extracted |
| `last_seen` | ISO 8601 | Most recent episode confirming this fact |
| `source_memory_ids` | JSON array | Memory IDs that contributed evidence |

The `(entity, attribute)` pair is unique — a second observation of the same fact strengthens confidence rather than creating a new row. Confidence update formula: `confidence += (1 − confidence) × 0.35` per additional observation.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/nimaabaeian/semantic-memory-agent.git
cd semantic-memory-agent
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

Get a key from [Google AI Studio](https://aistudio.google.com/).

### 3. Run the agent

```bash
python agent.py
```

Default behavior:
- Watches `./inbox/` for new episode files
- Generates episodic memories and semantic facts via Gemini LLM within 5 seconds of each file drop
- Consolidates cross-episode insights every 30 minutes
- Serves the memory query API at `http://localhost:8888`

### 4. Drop episode files into inbox

```bash
cp episode_001.json inbox/
cp session_02.txt   inbox/
cp audio_03.mp3     inbox/
cp video_04.mp4     inbox/
# Each file is picked up, ingested, and fact-extracted within 5 seconds
```

### 5. Ingest text directly via API

```bash
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Alex greeted the robot and asked for the weather.", "source": "manual"}'
```

If `source` ends in `.json`, the text is parsed as episode data. If it ends in `.txt`, it is framed as a transcript. Other sources receive light generic framing.

### 6. Query memories and facts

```bash
curl "http://localhost:8888/query?q=what+do+you+know+about+Alex"
```

### 7. Retrieve semantic facts

```bash
# All facts
curl http://localhost:8888/facts

# Facts about a specific person
curl "http://localhost:8888/facts?entity=Alex+Chen"
```

### 8. Check status

```bash
curl http://localhost:8888/status
# Returns: total_memories, unconsolidated, consolidations, facts
```

### 9. Trigger consolidation manually

```bash
curl -X POST http://localhost:8888/consolidate
```

### 10. Dashboard (optional)

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

## CLI Options

```
python agent.py [--watch DIR] [--port PORT] [--consolidate-every MIN]

  --watch DIR              Folder to watch for new files  (default: ./inbox)
  --port PORT              HTTP API port                  (default: 8888)
  --consolidate-every MIN  Consolidation interval         (default: 30)
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. Gemini API key. |
| `MODEL` | `gemini-3.1-flash-lite-preview` | Gemini model name. |
| `MEMORY_DB` | `semantic_memory.db` | SQLite database path. |

## API Reference

| Endpoint | Method | Body / Params | Description |
|---|---|---|---|
| `/status` | GET | — | Counts: `total_memories`, `unconsolidated`, `consolidations`, `facts` |
| `/memories` | GET | — | All episodic memories, most recent first (limit 50) |
| `/facts` | GET | `?entity=<name>` (optional) | Semantic facts sorted by confidence; partial entity name filter |
| `/query` | GET | `?q=<question>` | Natural language query across facts, memories, and consolidation history |
| `/ingest` | POST | `{"text": "...", "source": "..."}` | Ingest a text snippet directly |
| `/consolidate` | POST | — | Trigger consolidation immediately |
| `/delete` | POST | `{"memory_id": <int>}` | Delete one episodic memory by ID |
| `/clear` | POST | — | Delete all memories, facts, consolidations, processed-file records, **and all files in the inbox folder** |

## Project Structure

```
social-robot-semantic-memory-generator/
├── agent.py            # Core agent: file watcher, ingestion, fact extraction, consolidation, HTTP API
├── dashboard.py        # Streamlit UI (connects to the running agent API)
├── requirements.txt    # Python dependencies
├── inbox/              # Drop episode files here for auto-ingestion
├── docs/               # Assets
└── semantic_memory.db  # SQLite database (created automatically on first run)
```

## Consolidation

The consolidation agent runs on a background timer. It reads all unconsolidated memories (requires ≥ 2) and looks for:

- **Recurring people** — who appears across multiple episodes and what they typically do or say
- **Repeated interaction patterns** — common greetings, task types, conversation topics
- **Persistent preferences or habits** — food, activities, topics the person enjoys or avoids
- **Social tendencies** — whether someone is usually brief, talkative, task-focused, or emotionally expressive
- **Relationship development** — whether familiarity is increasing across episodes
- **Temporal or location patterns** — if present and consistent

Each consolidation produces one record with a synthesized summary, one actionable insight for the robot, and cross-memory connection links.

## Scope and Limitations

- **Memory generation only.** This agent ingests episode files and builds memories. It does not control robot behavior, publish to any robot middleware, or manage robot state.
- **No vector search.** Retrieval is full-table SQL. Works well at the scale of a single robot's interaction history; not designed for large multi-robot deployments.
- **20 MB inline media limit.** Imposed by Gemini's inline byte API. Large audio/video files must be trimmed before being dropped in inbox.
- **12,000 character text cap.** Text files read from inbox are truncated to 12,000 characters. A warning is logged when truncation occurs.
- **No episodic deduplication.** The same file dropped twice will be ingested once (tracked by path in `processed_files`), but semantically duplicate content from different filenames will produce separate memory entries. Facts, however, are deduplicated by `(entity, attribute)`.
- **Fact confidence is monotonically increasing.** Contradicting observations update the value but do not decrease confidence. There is no mechanism to reduce certainty when a person changes a preference.
- **Platform-agnostic.** The agent assumes the robot delivers episode files to `./inbox/`. Integration with any specific robot platform (ROS, NAOqi, etc.) is out of scope.

## Built With

- [Google ADK](https://google.github.io/adk-docs/) — agent orchestration and multi-agent routing
- [Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models) — LLM for memory generation, fact extraction, consolidation, and query
- [SQLite](https://www.sqlite.org/) — persistent memory store
- [aiohttp](https://docs.aiohttp.org/) — async HTTP API
- [Streamlit](https://streamlit.io/) — optional dashboard UI

## License

MIT
