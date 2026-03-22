"""
Agent Memory Layer — Always-On ADK Agent

A lightweight, cost-effective background agent that continuously processes, consolidates, and serves memory. Runs 24/7 on Gemini 3.1 Flash-Lite.

Usage:
    python agent.py                          # watch ./inbox, serve on :8888
    python agent.py --watch ./docs --port 9000
    python agent.py --consolidate-every 15   # consolidate every 15 min

Query:
    curl "http://localhost:8888/query?q=what+do+you+know"
    curl -X POST http://localhost:8888/ingest -d '{"text": "some info"}'
"""

import argparse
import asyncio
import json
import logging
import mimetypes
import os
import shutil
import signal
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ─── Config ────────────────────────────────────────────────────

MODEL = os.getenv("MODEL", "gemini-3.1-flash-lite-preview")
DB_PATH = os.getenv("MEMORY_DB", "semantic_memory.db")

# Supported file types for multimodal ingestion
#   JSON  → parsed as structured episode data with social-context framing
#   .txt  → treated as human-robot interaction transcripts when applicable
#   media → sent as multimodal bytes with social episode framing
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".yml"}
TRANSCRIPT_EXTENSIONS = {".txt"}  # plain-text files treated as interaction transcripts
MEDIA_EXTENSIONS = {
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    # Video
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    # Documents
    ".pdf": "application/pdf",
}
ALL_SUPPORTED = TEXT_EXTENSIONS | set(MEDIA_EXTENSIONS.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="[%H:%M]",
)
log = logging.getLogger("memory-agent")


# ─── Episode Helpers ───────────────────────────────────────────


def parse_json_episode(text: str, filename: str) -> str:
    """Convert a JSON episode file into a richly annotated text prompt.

    Handles dict-style single episodes and list-style episode arrays.
    Gracefully falls back to raw text if JSON is malformed.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text

    def describe(obj, indent: int = 0) -> str:
        """Recursively render a JSON object as readable indented prose."""
        pad = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.append(describe(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {v}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            items = []
            for i, item in enumerate(obj):
                prefix = f"{pad}[{i}] "
                if isinstance(item, (dict, list)):
                    items.append(prefix.rstrip())
                    items.append(describe(item, indent + 1))
                else:
                    items.append(f"{prefix}{item}")
            return "\n".join(items)
        else:
            return f"{pad}{obj}"

    structured = describe(data)
    header = (
        f"=== Social Robot Episode File: {filename} ===\n"
        "This JSON file contains structured data from a recorded human-robot interaction.\n"
        "Extract all socially meaningful information: who participated, what happened,\n"
        "what was said, the interaction outcome, any expressed preferences or opinions.\n\n"
    )
    return header + structured


def build_text_ingest_prompt(text: str, filename: str) -> str:
    """Return the right framing prompt for a text file based on its type."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".json":
        return parse_json_episode(text, filename)

    if suffix in TRANSCRIPT_EXTENSIONS:
        return (
            f"=== Interaction Transcript: {filename} ===\n"
            "This is a conversation transcript from a human-robot interaction.\n"
            "Focus on socially meaningful content: participant names, stated preferences\n"
            "or opinions, notable statements, emotional cues, and interaction dynamics.\n"
            "Capture durable personal details. Avoid storing trivial filler exchanges.\n\n"
            + text
        )

    # Generic text (markdown, log, etc.) — plain pass-through with light framing
    return f"Source: {filename}\n\n" + text


def build_media_ingest_prompt(filename: str, mime_type: str, size_mb: float) -> str:
    """Return a social-episode-aware prompt for multimodal media ingestion."""
    media_kind = mime_type.split("/")[0]  # 'audio' | 'video' | 'image'
    return (
        f"=== Social Robot Episode {media_kind.capitalize()} File: {filename} "
        f"({size_mb:.1f}MB) ===\n"
        f"This {media_kind} file is evidence from a human-robot interaction episode.\n"
        "Analyze it for:\n"
        "  - Participant identities and voices (if distinguishable)\n"
        "  - Emotional tone, engagement level, and social dynamics\n"
        "  - What was said or what activity occurred\n"
        "  - Key social moments, outcomes, or notable exchanges\n"
        "Extract all socially meaningful information and store it as an interaction memory.\n"
        f"Source file: {filename}, MIME: {mime_type}"
    )

# ─── Database ──────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL DEFAULT '',
            raw_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            entities TEXT NOT NULL DEFAULT '[]',
            topics TEXT NOT NULL DEFAULT '[]',
            connections TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS consolidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_ids TEXT NOT NULL,
            summary TEXT NOT NULL,
            insight TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS processed_files (
            path TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity TEXT NOT NULL,
            attribute TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            source_memory_ids TEXT NOT NULL DEFAULT '[]',
            UNIQUE(entity, attribute)
        );
    """)
    return db


# ─── ADK Tools ─────────────────────────────────────────────────


def store_memory(
    raw_text: str,
    summary: str,
    entities: list[str],
    topics: list[str],
    importance: float,
    source: str = "",
) -> dict:
    """Store a processed memory in the database.

    Args:
        raw_text: The original input text.
        summary: A concise 1-2 sentence summary.
        entities: Key people, companies, products, or concepts.
        topics: 2-4 topic tags.
        importance: Float 0.0 to 1.0 indicating importance.
        source: Where this memory came from (filename, URL, etc).

    Returns:
        dict with memory_id and confirmation.
    """
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = db.execute(
        """INSERT INTO memories (source, raw_text, summary, entities, topics, importance, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (source, raw_text, summary, json.dumps(entities), json.dumps(topics), importance, now),
    )
    db.commit()
    mid = cursor.lastrowid
    db.close()
    log.info(f"MEMORY stored #{mid} (importance={importance:.2f}): {summary[:80]}")
    return {"memory_id": mid, "status": "stored", "summary": summary}


def read_all_memories() -> dict:
    """Read all stored memories from the database, most recent first.

    Returns:
        dict with list of memories and count.
    """
    db = get_db()
    rows = db.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT 50").fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "source": r["source"], "raw_text": r["raw_text"],
            "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "connections": json.loads(r["connections"]),
            "created_at": r["created_at"], "consolidated": bool(r["consolidated"]),
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


def read_unconsolidated_memories() -> dict:
    """Read memories that haven't been consolidated yet.

    Returns:
        dict with list of unconsolidated memories and count.
    """
    db = get_db()
    rows = db.execute(
        "SELECT * FROM memories WHERE consolidated = 0 ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "created_at": r["created_at"],
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


def store_consolidation(
    source_ids: list[int],
    summary: str,
    insight: str,
    connections: list[dict],
) -> dict:
    """Store a consolidation result and mark source memories as consolidated.

    Args:
        source_ids: List of memory IDs that were consolidated.
        summary: A synthesized summary across all source memories.
        insight: One key pattern or insight discovered.
        connections: List of dicts with 'from_id', 'to_id', 'relationship'.

    Returns:
        dict with confirmation.
    """
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO consolidations (source_ids, summary, insight, created_at) VALUES (?, ?, ?, ?)",
        (json.dumps(source_ids), summary, insight, now),
    )
    for conn in connections:
        from_id, to_id = conn.get("from_id"), conn.get("to_id")
        rel = conn.get("relationship", "")
        if from_id and to_id:
            for mid in [from_id, to_id]:
                row = db.execute("SELECT connections FROM memories WHERE id = ?", (mid,)).fetchone()
                if row:
                    existing = json.loads(row["connections"])
                    existing.append({"linked_to": to_id if mid == from_id else from_id, "relationship": rel})
                    db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
    placeholders = ",".join("?" * len(source_ids))
    db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})", source_ids)
    db.commit()
    db.close()
    log.info(f"CONSOLIDATE stored: {len(source_ids)} memories merged. Insight: {insight[:100]}")
    return {"status": "consolidated", "memories_processed": len(source_ids), "insight": insight}


def read_consolidation_history() -> dict:
    """Read past consolidation insights.

    Returns:
        dict with list of consolidation records.
    """
    db = get_db()
    rows = db.execute("SELECT * FROM consolidations ORDER BY created_at DESC LIMIT 10").fetchall()
    result = [{"summary": r["summary"], "insight": r["insight"], "source_ids": r["source_ids"]} for r in rows]
    db.close()
    return {"consolidations": result, "count": len(result)}


def get_memory_stats() -> dict:
    """Get current memory statistics.

    Returns:
        dict with counts of memories, consolidations, etc.
    """
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    unconsolidated = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
    consolidations = db.execute("SELECT COUNT(*) as c FROM consolidations").fetchone()["c"]
    facts = db.execute("SELECT COUNT(*) as c FROM facts").fetchone()["c"]
    db.close()
    return {
        "total_memories": total,
        "unconsolidated": unconsolidated,
        "consolidations": consolidations,
        "facts": facts,
    }


def delete_memory(memory_id: int) -> dict:
    """Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        dict with status.
    """
    db = get_db()
    row = db.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"status": "not_found", "memory_id": memory_id}
    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    db.close()
    log.info(f"DELETE memory #{memory_id}")
    return {"status": "deleted", "memory_id": memory_id}


def clear_all_memories(inbox_path: str | None = None) -> dict:
    """Delete all memories, consolidations, and inbox files. Full reset."""
    db = get_db()
    mem_count = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    db.execute("DELETE FROM memories")
    db.execute("DELETE FROM consolidations")
    db.execute("DELETE FROM processed_files")
    db.execute("DELETE FROM facts")
    db.commit()
    db.close()

    # Also clear the inbox folder so files aren't re-ingested
    files_deleted = 0
    if inbox_path:
        folder = Path(inbox_path)
        if folder.is_dir():
            for f in folder.iterdir():
                if f.name.startswith("."):
                    continue  # keep hidden files like .gitkeep
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except OSError as e:
                    log.error(f"Failed to delete {f.name}: {e}")

    log.info(f"CLEAR all {mem_count} memories, {files_deleted} inbox files deleted")
    return {"status": "cleared", "memories_deleted": mem_count, "files_deleted": files_deleted}


def upsert_fact(entity: str, attribute: str, value: str, memory_id: int = 0) -> dict:
    """Insert a new semantic fact or strengthen an existing one.

    Each time the same (entity, attribute) is observed again, confidence increases
    toward 1.0 and evidence_count increments. The value is updated to the latest
    observation.

    Args:
        entity: The person, place, or thing this fact is about (use full name).
        attribute: Snake_case key for the aspect being described
                   (e.g. preferred_speech_speed, coffee_preference, visit_pattern).
        value: The observed value as a concise string.
        memory_id: ID of the source memory (0 if unknown).

    Returns:
        dict with status ('created' or 'strengthened'), entity, attribute, confidence,
        and evidence_count.
    """
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = db.execute(
        "SELECT * FROM facts WHERE entity = ? AND attribute = ?", (entity, attribute)
    ).fetchone()

    if row:
        new_confidence = min(1.0, row["confidence"] + (1.0 - row["confidence"]) * 0.35)
        new_count = row["evidence_count"] + 1
        ids = json.loads(row["source_memory_ids"])
        if memory_id and memory_id not in ids:
            ids.append(memory_id)
        db.execute(
            """UPDATE facts SET value = ?, confidence = ?, evidence_count = ?,
               last_seen = ?, source_memory_ids = ? WHERE entity = ? AND attribute = ?""",
            (value, new_confidence, new_count, now, json.dumps(ids), entity, attribute),
        )
        db.commit()
        db.close()
        log.info(f"FACT strengthened [{entity}] {attribute}={value!r} (confidence={new_confidence:.2f}, n={new_count})")
        return {"status": "strengthened", "entity": entity, "attribute": attribute,
                "value": value, "confidence": new_confidence, "evidence_count": new_count}
    else:
        ids = [memory_id] if memory_id else []
        db.execute(
            """INSERT INTO facts (entity, attribute, value, confidence, evidence_count,
               first_seen, last_seen, source_memory_ids) VALUES (?, ?, ?, 0.5, 1, ?, ?, ?)""",
            (entity, attribute, value, now, now, json.dumps(ids)),
        )
        db.commit()
        db.close()
        log.info(f"FACT created [{entity}] {attribute}={value!r} (confidence=0.50)")
        return {"status": "created", "entity": entity, "attribute": attribute,
                "value": value, "confidence": 0.5, "evidence_count": 1}


def read_facts(entity: str = "") -> dict:
    """Read semantic facts from the database, optionally filtered by entity.

    Args:
        entity: Filter by entity name (partial, case-insensitive). Empty returns all.

    Returns:
        dict with list of facts sorted by confidence descending.
    """
    db = get_db()
    if entity:
        rows = db.execute(
            "SELECT * FROM facts WHERE entity LIKE ? ORDER BY confidence DESC LIMIT 100",
            (f"%{entity}%",),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM facts ORDER BY confidence DESC LIMIT 100"
        ).fetchall()
    facts = [
        {
            "id": r["id"], "entity": r["entity"], "attribute": r["attribute"],
            "value": r["value"], "confidence": r["confidence"],
            "evidence_count": r["evidence_count"],
            "first_seen": r["first_seen"], "last_seen": r["last_seen"],
        }
        for r in rows
    ]
    db.close()
    return {"facts": facts, "count": len(facts)}


# ─── ADK Agents ────────────────────────────────────────────────


def build_agents():
    ingest_agent = Agent(
        name="ingest_agent",
        model=MODEL,
        description="Processes episode data from human-robot interactions into structured memory.",
        instruction=(
            "You are a Social Episode Memory Agent for a social robot.\n"
            "You process episode data from human-robot interactions: JSON episode files,\n"
            "conversation transcripts, audio recordings, and video recordings.\n\n"
            "For ANY input you receive, follow these steps:\n"
            "1. Write a full description of what happened in this social interaction.\n"
            "2. Write a concise 1–2 sentence summary focused on participants and social outcome.\n"
            "3. Extract entities: full names of people, locations, objects, or concepts central\n"
            "   to the interaction. Prefer specific names over generic labels.\n"
            "4. Assign 2–4 topic tags that describe the interaction type and themes\n"
            "   (e.g. 'greeting', 'task-request', 'preference-expressed', 'farewell',\n"
            "    'emotional-support', 'information-seeking', 'shared-activity').\n"
            "5. Rate social importance 0.0–1.0 using this scale:\n"
            "   • 0.8–1.0: strong emotional content, first meeting, explicit preference or\n"
            "     dislike, conflict, notable achievement, personal disclosure\n"
            "   • 0.5–0.7: routine but meaningful interaction, clear task completion,\n"
            "     sustained conversation with some personal content\n"
            "   • 0.2–0.4: brief or uneventful exchange with little memorable content\n"
            "6. Call store_memory with raw_text=full description, summary, entities,\n"
            "   topics, importance, and source.\n\n"
            "Focus your extraction on:\n"
            "  - Who was involved and their relationship to the robot\n"
            "  - What happened and how the interaction unfolded\n"
            "  - What people said, especially preferences, opinions, personal details\n"
            "  - The social outcome or emotional result of the interaction\n"
            "  - Any recurring patterns, habits, or preferences clearly demonstrated\n\n"
            "For JSON episode files: use the structured fields to build a rich social description.\n"
            "For transcripts: emphasize names, stated preferences, notable statements, dynamics.\n"
            "For audio/video: describe voices, emotional tone, activity, and key social moments.\n\n"
            "Always call store_memory. After storing, confirm in one sentence what was captured."
        ),
        tools=[store_memory],
    )

    fact_extractor_agent = Agent(
        name="fact_extractor_agent",
        model=MODEL,
        description="Extracts atemporal semantic facts from a freshly ingested episode memory.",
        instruction=(
            "You are a Semantic Fact Extractor for a social robot.\n"
            "You receive the content of a social interaction episode that was just stored.\n"
            "Your job is to extract only facts that are ATEMPORAL — things that will remain\n"
            "true beyond this specific episode and are useful for future interactions.\n\n"
            "A semantic fact has three parts:\n"
            "  entity    — the specific named person, place, or thing (use full name if known)\n"
            "  attribute — a short snake_case descriptor of what this fact is about,\n"
            "              e.g. preferred_speech_speed, coffee_preference, visit_pattern,\n"
            "                   communication_style, role, typical_mood, disliked_topics\n"
            "  value     — the observed value as a concise string\n\n"
            "Only extract facts that are:\n"
            "  • About a SPECIFIC named entity (not 'the user' or 'a person')\n"
            "  • Likely to be STABLE or REPEATING across future interactions\n"
            "  • Expressed or clearly demonstrated (not inferred from a single ambiguous cue)\n"
            "  • Useful for the robot to remember (preferences, habits, communication style,\n"
            "    role, relationship to the robot, recurring patterns)\n\n"
            "Do NOT extract:\n"
            "  • One-off events ('asked for weather today', 'arrived late on Tuesday')\n"
            "  • Timestamps, dates, or episode-specific details\n"
            "  • Vague or low-confidence observations\n"
            "  • Facts about the robot itself\n\n"
            "For each fact you identify, call upsert_fact(entity, attribute, value, memory_id).\n"
            "Pass memory_id=0 if the episode memory ID is not available.\n"
            "If no durable facts can be extracted, respond: 'No durable facts in this episode.'\n"
            "After calling upsert_fact for each fact, briefly list what was stored."
        ),
        tools=[upsert_fact],
    )

    consolidate_agent = Agent(
        name="consolidate_agent",
        model=MODEL,
        description="Consolidates interaction memories into robot-useful social insights.",
        instruction=(
            "You are a Social Memory Consolidation Agent for a social robot.\n"
            "Your job is to synthesize episodic interaction memories into durable insights\n"
            "that help the robot behave more appropriately in future interactions.\n\n"
            "Steps:\n"
            "1. Call read_unconsolidated_memories to retrieve pending memories.\n"
            "2. If fewer than 2 memories exist, respond: 'Nothing to consolidate yet.'\n"
            "3. Analyze memories for cross-episode social patterns:\n"
            "   • Recurring people: who appears often, what they typically do or say\n"
            "   • Repeated interaction patterns: common greetings, task requests, topics\n"
            "   • Persistent preferences or habits: food, activities, topics they enjoy/avoid\n"
            "   • Social tendencies: is someone usually cheerful, task-focused, brief, talkative?\n"
            "   • Location or temporal patterns if present\n"
            "   • Relationship development: is familiarity increasing across episodes?\n"
            "4. Write a concise synthesized summary across all source memories.\n"
            "5. Identify ONE key actionable insight for the robot — something concrete it can\n"
            "   use in future interactions (e.g., 'User prefers brief responses',\n"
            "   'Alex typically initiates with a task then transitions to small talk',\n"
            "   'Morning interactions are usually shorter than afternoon ones').\n"
            "6. Call store_consolidation with source_ids, summary, insight, and connections.\n\n"
            "Connections: list of dicts with 'from_id', 'to_id', 'relationship' keys.\n"
            "Link memories where the same person, preference, or pattern recurs.\n"
            "Prioritize connections that reveal social tendencies over coincidental links."
        ),
        tools=[read_unconsolidated_memories, store_consolidation],
    )

    query_agent = Agent(
        name="query_agent",
        model=MODEL,
        description="Answers questions using stored memories.",
        instruction=(
            "You are a Memory Query Agent. When asked a question:\n"
            "1. Call read_facts to retrieve atemporal semantic facts (what is persistently known about people)\n"
            "2. Call read_all_memories to access the full episodic memory store\n"
            "3. Call read_consolidation_history for cross-episode insights\n"
            "4. Synthesize an answer drawing from all three sources\n"
            "5. Prefer facts for stable truths ('Alex prefers short responses');\n"
            "   use memories for context, events, and specifics\n"
            "6. Cite sources: [Fact: entity/attribute], [Memory N], [Consolidation N]\n"
            "7. If no relevant data exists, say so honestly\n\n"
            "Be thorough but concise. Always cite sources."
        ),
        tools=[read_all_memories, read_consolidation_history, read_facts],
    )

    orchestrator = Agent(
        name="memory_orchestrator",
        model=MODEL,
        description="Routes memory operations to specialist agents.",
        instruction=(
            "You are the Memory Orchestrator for an always-on memory system.\n"
            "Route requests to the right sub-agent:\n"
            "- New information → ALWAYS run BOTH steps in order:\n"
            "  1. ingest_agent — stores the episode and returns a summary + memory_id\n"
            "  2. fact_extractor_agent — pass it the text summary/description that\n"
            "     ingest_agent produced (not raw media bytes); it will extract durable facts\n"
            "- Consolidation request → consolidate_agent\n"
            "- Questions → query_agent\n"
            "- Status check → call get_memory_stats and report\n\n"
            "Never skip fact_extractor_agent after an ingest. It runs on every episode.\n"
            "After both steps complete, give a one-sentence summary of what was stored."
        ),
        sub_agents=[ingest_agent, fact_extractor_agent, consolidate_agent, query_agent],
        tools=[get_memory_stats],
    )

    return orchestrator


# ─── Agent Runner ──────────────────────────────────────────────


class MemoryAgent:
    def __init__(self):
        self.agent = build_agents()
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="memory_layer",
            session_service=self.session_service,
        )

    async def run(self, message: str) -> str:
        session = await self.session_service.create_session(
            app_name="memory_layer", user_id="agent",
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
        return await self._execute(session, content)

    async def run_multimodal(self, text: str, file_bytes: bytes, mime_type: str) -> str:
        """Send a multimodal message with both text and a media file."""
        session = await self.session_service.create_session(
            app_name="memory_layer", user_id="agent",
        )
        parts = [
            types.Part.from_text(text=text),
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        ]
        content = types.Content(role="user", parts=parts)
        return await self._execute(session, content)

    async def _execute(self, session, content: types.Content) -> str:
        """Run the agent with the given content and return the text response."""
        response = ""
        async for event in self.runner.run_async(
            user_id="agent", session_id=session.id, new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response += part.text
        return response

    async def ingest(self, text: str, source: str = "") -> str:
        msg = f"Remember this information (source: {source}):\n\n{text}" if source else f"Remember this information:\n\n{text}"
        return await self.run(msg)

    async def ingest_file(self, file_path: Path) -> str:
        """Ingest a media file (audio, video, image, PDF) as a multimodal episode."""
        suffix = file_path.suffix.lower()
        mime_type = MEDIA_EXTENSIONS.get(suffix)
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"

        file_bytes = file_path.read_bytes()
        size_mb = len(file_bytes) / (1024 * 1024)

        # Gemini has a ~20MB inline limit; skip very large files
        if size_mb > 20:
            log.warning(f"SKIP {file_path.name} ({size_mb:.1f} MB) — exceeds 20 MB inline limit")
            return f"Skipped: file too large ({size_mb:.1f}MB)"

        media_kind = mime_type.split("/")[0]
        prompt = build_media_ingest_prompt(file_path.name, mime_type, size_mb)
        log.info(f"INGEST [{media_kind}] {file_path.name} ({size_mb:.1f} MB)")
        return await self.run_multimodal(prompt, file_bytes, mime_type)

    async def consolidate(self) -> str:
        return await self.run("Consolidate unconsolidated memories. Find connections and patterns.")

    async def query(self, question: str) -> str:
        return await self.run(f"Based on my memories, answer: {question}")

    async def status(self) -> str:
        return await self.run("Give me a status report on my memory system.")


# ─── File Watcher ──────────────────────────────────────────────


async def watch_folder(agent: MemoryAgent, folder: Path, poll_interval: int = 5):
    """Watch inbox for new episode files and ingest them with type-appropriate framing.

    File routing:
      .json        → parsed as structured social robot episode data
      .txt / .md   → treated as human-robot interaction transcripts
      audio/video  → sent multimodal with social episode framing
      other text   → standard text ingestion
    """
    folder.mkdir(parents=True, exist_ok=True)
    db = get_db()
    log.info(f"WATCH {folder}/")

    while True:
        try:
            for f in sorted(folder.iterdir()):
                if f.name.startswith("."):
                    continue
                suffix = f.suffix.lower()
                if suffix not in ALL_SUPPORTED:
                    continue
                if db.execute("SELECT 1 FROM processed_files WHERE path = ?", (str(f),)).fetchone():
                    continue

                try:
                    if suffix in TEXT_EXTENSIONS:
                        raw_full = f.read_text(encoding="utf-8", errors="replace")
                        raw = raw_full[:12000]
                        if len(raw_full) > 12000:
                            log.warning(f"TRUNCATE {f.name}: {len(raw_full)} chars → 12,000 (content beyond limit dropped)")
                        if not raw.strip():
                            log.warning(f"SKIP {f.name} — empty file")
                        else:
                            file_type = "json-episode" if suffix == ".json" else (
                                "transcript" if suffix in TRANSCRIPT_EXTENSIONS else "text"
                            )
                            log.info(f"INGEST [{file_type}] {f.name}")
                            prompt = build_text_ingest_prompt(raw, f.name)
                            await agent.ingest(prompt)
                    else:
                        await agent.ingest_file(f)
                except Exception as file_err:
                    log.error(f"ERROR ingesting {f.name}: {file_err}")

                db.execute(
                    "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                    (str(f), datetime.now(timezone.utc).isoformat()),
                )
                db.commit()
        except Exception as e:
            log.error(f"WATCH error: {e}")

        await asyncio.sleep(poll_interval)


# ─── Consolidation Timer ──────────────────────────────────────


async def consolidation_loop(agent: MemoryAgent, interval_minutes: int = 30):
    """Run social memory consolidation periodically to surface cross-episode insights."""
    log.info(f"CONSOLIDATE every {interval_minutes} min")
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            db = get_db()
            count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
            db.close()
            if count >= 2:
                log.info(f"CONSOLIDATE running ({count} unconsolidated memories)")
                result = await agent.consolidate()
                log.info(f"CONSOLIDATE done — {result[:120]}".rstrip())
            else:
                log.info(f"CONSOLIDATE skipped ({count} unconsolidated memories — need ≥ 2)")
        except Exception as e:
            log.error(f"CONSOLIDATE error: {e}")


# ─── HTTP API ──────────────────────────────────────────────────


def build_http(agent: MemoryAgent, watch_path: str = "./inbox"):
    app = web.Application()

    async def handle_query(request: web.Request):
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        answer = await agent.query(q)
        return web.json_response({"question": q, "answer": answer})

    async def handle_ingest(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        source = data.get("source", "api")
        framed = build_text_ingest_prompt(text, source)
        result = await agent.ingest(framed)
        return web.json_response({"status": "ingested", "response": result})

    async def handle_consolidate(request: web.Request):
        result = await agent.consolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_status(request: web.Request):
        stats = get_memory_stats()
        return web.json_response(stats)

    async def handle_memories(request: web.Request):
        data = read_all_memories()
        return web.json_response(data)

    async def handle_delete(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        memory_id = data.get("memory_id")
        if not memory_id:
            return web.json_response({"error": "missing 'memory_id' field"}, status=400)
        result = delete_memory(int(memory_id))
        return web.json_response(result)

    async def handle_facts(request: web.Request):
        entity = request.query.get("entity", "").strip()
        data = read_facts(entity=entity)
        return web.json_response(data)

    async def handle_clear(request: web.Request):
        result = clear_all_memories(inbox_path=watch_path)
        return web.json_response(result)

    app.router.add_get("/query", handle_query)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_post("/consolidate", handle_consolidate)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/memories", handle_memories)
    app.router.add_get("/facts", handle_facts)
    app.router.add_post("/delete", handle_delete)
    app.router.add_post("/clear", handle_clear)

    return app


# ─── Main ──────────────────────────────────────────────────────


async def main_async(args):
    agent = MemoryAgent()

    log.info("Social Robot Episode Memory Agent starting")
    log.info(f"  Model    : {MODEL}")
    log.info(f"  Database : {DB_PATH}")
    log.info(f"  Inbox    : {args.watch}")
    log.info(f"  Consolidate every {args.consolidate_every} min")
    log.info(f"  API      : http://localhost:{args.port}")
    log.info("")

    # Start background tasks
    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
    ]

    # Start HTTP server
    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info(f"Ready — drop episode files in {args.watch}/ or POST to http://localhost:{args.port}/ingest")
    log.info("  Accepted: .json (episode), .txt (transcript), audio, video, image, PDF")
    log.info("")

    # Wait forever
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Layer - Always-On ADK Agent")
    parser.add_argument("--watch", default="./inbox", help="Folder to watch for new files (default: ./inbox)")
    parser.add_argument("--port", type=int, default=8888, help="HTTP API port (default: 8888)")
    parser.add_argument("--consolidate-every", type=int, default=30, help="Consolidation interval in minutes (default: 30)")
    args = parser.parse_args()

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown(sig):
        log.info(f"\n👋 Shutting down (signal {sig})...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()
        log.info("Agent stopped.")


if __name__ == "__main__":
    main()
