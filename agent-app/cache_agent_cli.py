#!/usr/bin/env python3
"""Standalone cache agent CLI.

Goal: quickly test natural-language queries against the local SQLite cache.

Strategy:
- If OPENAI_API_KEY is set (and --no-openai is NOT used), ask the model to pick a
  query strategy and parameters.
- Execute only pre-defined, safe SQL templates (no arbitrary SQL).
- Print results in a compact, test-friendly format.

This intentionally does NOT depend on the web server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any

import requests

try:
    from env_loader import load_env

    load_env()
except Exception:
    pass


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install via APT: sudo apt install python3-yaml\n"
            f"Import error: {e}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _cfg_get(cfg: dict, *keys: str, default=None):
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _resolve_sqlite_path(cfg_path: str, cfg: dict, override_db: str | None) -> str:
    if override_db:
        return os.path.abspath(override_db)

    raw = _cfg_get(cfg, "storage", "sqlite_path", default=None)
    if not raw:
        raise RuntimeError("storage.sqlite_path missing in config (or pass --db).")

    p = str(raw)
    if os.path.isabs(p):
        return p

    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    return os.path.abspath(os.path.join(cfg_dir, p))


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso_z(d: dt.datetime) -> str:
    return d.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str) -> dt.datetime | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _tail_text_file(path: str, *, max_bytes: int = 16384) -> str:
    try:
        p = (path or "").strip()
        if not p or not os.path.exists(p):
            return ""
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _is_sync_request(q: str) -> bool:
    s = (q or "").strip().lower()
    if not s:
        return False
    # Keep this intentionally conservative.
    return bool(
        re.search(
            r"\b(sync|synchronize|synchronise|refresh|update)\b(?:\s+(?:the\s+)?(cache|mail|mailbox))?\b",
            s,
        )
        or s in {"sync", "sync now", "sync cache", "update cache", "refresh cache"}
    )


def _is_sync_status_request(q: str) -> bool:
    s = (q or "").strip().lower()
    if not s:
        return False
    return bool(
        re.search(r"\b(sync|synchronization|synchronisation)\b.*\b(status|progress|running|log)\b", s)
        or s in {"sync status", "sync progress", "sync running"}
    )


def _is_cache_status_request(q: str) -> bool:
    s = (q or "").strip().lower()
    if not s:
        return False
    return bool(re.search(r"\b(cache)\b.*\b(status|coverage|range|info|what\s+is\s+in)\b", s))


def _sync_state_get(cfg: dict) -> dict:
    sa = cfg.get("search_agent")
    if not isinstance(sa, dict):
        return {}
    s = sa.get("sync")
    return s if isinstance(s, dict) else {}


def _sync_state_set(cfg: dict, state: dict):
    if "search_agent" not in cfg or not isinstance(cfg.get("search_agent"), dict):
        cfg["search_agent"] = {}
    cfg["search_agent"]["sync"] = state


def _start_full_sync(cfg_path: str, cfg: dict) -> dict:
    # If a previous run is still in progress, do not start another.
    st = _sync_state_get(cfg)
    try:
        pid0 = int(st.get("pid") or 0)
    except Exception:
        pid0 = 0
    if pid0 and _pid_alive(pid0):
        return {
            "ok": True,
            "action": "sync_status",
            "running": True,
            "pid": pid0,
            "log_path": str(st.get("log_path") or ""),
            "started_at": str(st.get("started_at") or ""),
            "note": "sync already running",
        }

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = os.path.join("/tmp", f"mailboxsync_agent_{ts}.log")
    cmd = [
        "python3",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sync-app", "mailboxsync.py")),
        "--config",
        os.path.abspath(cfg_path),
        "sync",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_COLOR"] = "1"

    try:
        fh = open(log_path, "ab", buffering=0)
    except Exception as e:
        return {"ok": False, "action": "sync_start_failed", "error": f"failed to open log: {e}"}

    try:
        p = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except Exception as e:
        try:
            fh.close()
        except Exception:
            pass
        return {"ok": False, "action": "sync_start_failed", "error": f"failed to start sync: {e}"}

    state = {
        "pid": int(p.pid),
        "log_path": log_path,
        "started_at": _iso_z(_utc_now()),
        "last_action": "start",
    }
    _sync_state_set(cfg, state)

    return {
        "ok": True,
        "action": "sync_started",
        "running": True,
        "pid": int(p.pid),
        "log_path": log_path,
        "started_at": state["started_at"],
    }


def _sync_status(cfg: dict) -> dict:
    st = _sync_state_get(cfg)
    try:
        pid = int(st.get("pid") or 0)
    except Exception:
        pid = 0
    log_path = str(st.get("log_path") or "")
    started_at = str(st.get("started_at") or "")
    running = bool(pid and _pid_alive(pid))
    tail = _tail_text_file(log_path, max_bytes=12000) if log_path else ""
    return {
        "ok": True,
        "action": "sync_status",
        "running": running,
        "pid": pid,
        "log_path": log_path,
        "started_at": started_at,
        "log_tail": tail[-8000:] if tail else "",
    }


def _cache_bounds(con: sqlite3.Connection) -> tuple[dt.datetime | None, dt.datetime | None]:
    """Best-effort cache coverage bounds (UTC)."""
    try:
        row = con.execute("SELECT min(start_dt) AS s, max(end_dt) AS e FROM coverage").fetchone()
        if row and (row[0] or row[1]):
            s = _parse_iso_utc(str(row[0] or ""))
            e = _parse_iso_utc(str(row[1] or ""))
            return (s, e)
    except Exception:
        pass

    # Fallback: infer from messages.
    try:
        row = con.execute("SELECT min(received_dt) AS s, max(received_dt) AS e FROM messages").fetchone()
        if row and (row[0] or row[1]):
            s = _parse_iso_utc(str(row[0] or ""))
            e = _parse_iso_utc(str(row[1] or ""))
            return (s, e)
    except Exception:
        pass
    return (None, None)


def _chat_model(cfg: dict, override: str | None) -> str:
    if override:
        return override
    m = str(_cfg_get(cfg, "chat", "model", default="gpt-5.1") or "gpt-5.1").strip()
    return m or "gpt-5.1"


def _openai_key() -> str | None:
    k = os.environ.get("OPENAI_API_KEY")
    return k.strip() if isinstance(k, str) and k.strip() else None


def _openai_plan(
    *,
    model: str,
    message: str,
    selected_message_id: str | None,
    previous_response_id: str | None,
) -> dict:
    """Ask OpenAI to choose a query strategy.

    Returns a dict plan. This is best-effort; caller should validate fields.
    """

    key = _openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    system = (
        "You are a local mailbox cache agent. "
        "You know the SQLite schema and can choose a safe query strategy. "
        "Return ONLY valid JSON (no markdown)."
    )

    schema = {
        "tables": {
            "messages": [
                "graph_id",
                "folder_id",
                "received_dt",
                "sent_dt",
                "subject",
                "from_name",
                "from_address",
                "to_json",
                "cc_json",
                "body_text",
                "body_html",
                "has_attachments",
            ],
            "attachments": [
                "message_graph_id",
                "attachment_id",
                "name",
                "content_type",
                "size",
                "is_inline",
                "content_text",
                "downloaded",
                "error",
            ],
            "folders": ["folder_id", "folder_path"],
            "fts_messages": ["subject", "body_text"],
            "fts_attachments": ["name", "content_text"],
        }
    }

    allowed_strategies = [
        "search_messages",
        "search_attachments",
        "pdf_attachments",
        "messages_on_day",
        "messages_in_last_days",
        "selected_attachments",
        "summarize_selected_attachments",
        "sync_now",
        "sync_status",
        "cache_status",
    ]

    user = {
        "query": message,
        "selected_message_id": selected_message_id or "",
        "allowed_strategies": allowed_strategies,
        "schema": schema,
        "notes": [
            "Prefer `pdf_attachments` for requests like: 'pdfs from Annie', optionally with 'last N days'.",
            "Prefer `messages_on_day` for requests like: 'emails I got today' / 'emails from yesterday'.",
            "Prefer `messages_in_last_days` for requests like: 'emails from the last 30 days' / 'past week'.",
            "IMPORTANT: Temporal words like 'today', 'tomorrow', 'yesterday', 'last week' refer to received date filtering, not full-text search terms.",
            "Use `selected_attachments` or `summarize_selected_attachments` for: 'summarize the attachments on this email'.",
            "Use `search_attachments` for: 'insurance in any attachment'.",
            "Use `search_messages` for general email subject/body queries.",
            "If the request is ambiguous, still pick one strategy and include a brief rationale.",
        ],
        "output_format": {
            "strategy": "one of allowed_strategies",
            "params": "object of parameters for that strategy",
            "limit": "int",
            "rationale": "short string",
        },
        "strategy_params_schema": {
            "search_messages": {"q": "string"},
            "search_attachments": {"q": "string"},
            "pdf_attachments": {
                "days": "int|null (optional)",
                "sender": "string (optional sender name/email fragment)",
            },
            "messages_on_day": {
                "day": "string (required: today|yesterday|tomorrow|YYYY-MM-DD)",
                "sender": "string (optional sender name/email fragment)",
                "date_field": "string (optional: received|sent)",
            },
            "messages_in_last_days": {
                "days": "int (required; 0-365)",
                "sender": "string (optional sender name/email fragment)",
                "date_field": "string (optional: received|sent)",
            },
            "selected_attachments": {"message_id": "string (required; use selected_message_id if present)"},
            "summarize_selected_attachments": {"message_id": "string (required; use selected_message_id if present)"},
        },
    }

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:600]}")
    data = r.json() if r.content else {}

    response_id = str(data.get("id") or "").strip()

    out_texts: list[str] = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                out_texts.append(c.get("text", ""))

    text = "\n".join(out_texts).strip()
    if not text:
        raise RuntimeError("OpenAI returned empty output")

    # Expect strict JSON. If the model adds stray text, try extracting the first JSON object.
    try:
        plan = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if not m:
            raise RuntimeError(f"Could not parse JSON from model output: {text[:400]}")
        plan = json.loads(m.group(0))

    if isinstance(plan, dict) and response_id and not plan.get("response_id"):
        plan = dict(plan)
        plan["response_id"] = response_id
    return plan


def _write_yaml_config(path: str, cfg: dict):
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to write config.yaml. Install via APT: sudo apt install python3-yaml\n"
            f"Import error: {e}"
        )

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.replace(tmp, path)


def _agent_state_get(cfg: dict) -> dict:
    s = cfg.get("search_agent")
    return s if isinstance(s, dict) else {}


def _agent_state_set(cfg: dict, updates: dict) -> None:
    if "search_agent" not in cfg or not isinstance(cfg.get("search_agent"), dict):
        cfg["search_agent"] = {}
    st = cfg["search_agent"]
    if isinstance(st, dict):
        st.update(updates)


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _fts_query_from_text(q: str) -> str:
    # Minimal/robust: quote tokens, OR them.
    toks = re.findall(r"[A-Za-z0-9@._+'\-]{3,}", (q or "").strip())
    toks = [t.replace('"', "") for t in toks][:8]
    if not toks:
        return '"' + (q or "").replace('"', "") + '"'
    return " OR ".join([('"' + t + '"') for t in toks])


def _extract_days_hint(q: str) -> int | None:
    s = (q or "").lower()
    if "yesterday" in s:
        return 1
    m = re.search(r"\b(last|past)\s+(\d{1,3})\s+day", s)
    if m:
        try:
            return max(0, min(365, int(m.group(2))))
        except Exception:
            return None
    if "last week" in s or "past week" in s:
        return 7
    if "last month" in s or "past month" in s:
        return 30
    if "last year" in s or "past year" in s:
        return 365
    return None


def _extract_sender_hint(q: str) -> str | None:
    # Very light-touch hinting; not a full parser.
    s = (q or "").strip()
    m = re.search(r"\bfrom\b\s+(.+)$", s, flags=re.IGNORECASE)
    if not m:
        return None
    v = m.group(1).strip()
    # Strip common trailing time phrases so the sender token stays clean.
    v = re.sub(r"\b(last|past)\s+\d{1,3}\s+days?\b\s*$", "", v, flags=re.IGNORECASE).strip()
    v = re.sub(r"\b(last|past)\s+week\b\s*$", "", v, flags=re.IGNORECASE).strip()
    v = re.sub(r"\byesterday\b\s*$", "", v, flags=re.IGNORECASE).strip()
    return v[:80] if v else None


def _extract_date_field_hint(q: str) -> str:
    """Return 'sent' or 'received' based on query wording."""
    s = (q or "").lower()
    # If the user says sent, treat temporal intent as sent-date.
    if re.search(r"\b(sent|send|sent items)\b", s):
        return "sent"
    # If the user says got/received, treat as received-date.
    if re.search(r"\b(got|received|inbox)\b", s):
        return "received"
    return "received"


def _parse_dt_iso_z(s: str) -> dt.datetime | None:
    try:
        v = (s or "").strip()
        if not v:
            return None
        # SQLite cache typically stores UTC with a trailing Z.
        if v.endswith("Z") and "+" not in v:
            v = v[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(v)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _human_received(received_iso: str) -> tuple[str, int | None]:
    d = _parse_dt_iso_z(received_iso)
    if not d:
        return ((received_iso or "").strip(), None)
    now = dt.datetime.now(dt.timezone.utc)
    days_ago = (now.date() - d.date()).days
    if days_ago < 0:
        days_ago = 0
    stamp = d.strftime("%b %d, %Y %H:%M UTC")
    return (f"{stamp} ({days_ago} day{'s' if days_ago != 1 else ''} ago)", days_ago)


def _day_bounds_utc(day: str) -> tuple[str, str] | None:
    """Return (start_iso, end_iso) for a UTC calendar day.

    Supported day values:
    - 'today'
    - 'yesterday'
    - 'YYYY-MM-DD'
    """
    try:
        d = (day or "").strip().lower()
        now = _utc_now()
        if d in {"today", "now"}:
            start = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
            end = start + dt.timedelta(days=1)
            return (_iso_z(start), _iso_z(end))
        if d == "yesterday":
            end = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
            start = end - dt.timedelta(days=1)
            return (_iso_z(start), _iso_z(end))
        if d == "tomorrow":
            start = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
            end = start + dt.timedelta(days=1)
            return (_iso_z(start), _iso_z(end))

        # YYYY-MM-DD
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", d)
        if m:
            y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start = dt.datetime(y, mo, da, tzinfo=dt.timezone.utc)
            end = start + dt.timedelta(days=1)
            return (_iso_z(start), _iso_z(end))
    except Exception:
        return None
    return None


def _extract_temporal_intent(q: str) -> dict | None:
    """Return a date-filter plan override when query contains temporal intent.

    Assumption (per user preference): temporal words refer to received/sent time, not literal text search.
    """
    s = (q or "").strip().lower()
    if not s:
        return None

    date_field = _extract_date_field_hint(q)

    # Direct day words.
    if "today" in s:
        return {
            "strategy": "messages_on_day",
            "params": {"day": "today", "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }
    if "yesterday" in s:
        return {
            "strategy": "messages_on_day",
            "params": {"day": "yesterday", "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }
    if "tomorrow" in s:
        return {
            "strategy": "messages_on_day",
            "params": {"day": "tomorrow", "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }

    # Past ranges.
    days = _extract_days_hint(q)
    if days is not None:
        return {
            "strategy": "messages_in_last_days",
            "params": {"days": days, "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }

    # Future single-day: "in N days" / "N days from now" / "a week from now".
    m = re.search(r"\b(?:in\s+(\d{1,3})\s+days?|(?:\d{1,3})\s+days?\s+from\s+now)\b", s)
    if m:
        # If pattern was 'in N days', group(1) captures N; otherwise parse the leading number.
        n_raw = m.group(1) or re.search(r"\b(\d{1,3})\s+days?\b", m.group(0)).group(1)  # type: ignore[union-attr]
        try:
            n = max(0, min(365, int(n_raw)))
            target = (_utc_now() + dt.timedelta(days=n)).date().isoformat()
            return {
                "strategy": "messages_on_day",
                "params": {"day": target, "sender": _extract_sender_hint(q) or "", "date_field": date_field},
            }
        except Exception:
            return None

    if "next week" in s:
        target = (_utc_now() + dt.timedelta(days=7)).date().isoformat()
        return {
            "strategy": "messages_on_day",
            "params": {"day": target, "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }
    if "next month" in s:
        target = (_utc_now() + dt.timedelta(days=30)).date().isoformat()
        return {
            "strategy": "messages_on_day",
            "params": {"day": target, "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }

    # If it contains other strong temporal cues, treat as a date query for recent mail.
    if any(k in s for k in ("last ", "past ", "this week", "this month", "this year", "next ")):
        # Conservative default: last 7 days.
        return {
            "strategy": "messages_in_last_days",
            "params": {"days": 7, "sender": _extract_sender_hint(q) or "", "date_field": date_field},
        }
    return None


def _print_rows(rows: list[dict], *, title: str):
    print("\n==", title, "==")
    if not rows:
        print("(no results)")
        return
    for i, r in enumerate(rows, 1):
        # Keep stable keys first.
        mid = str(r.get("id") or r.get("graph_id") or "").strip()
        subj = str(r.get("subject") or "").strip()
        received = str(r.get("received") or r.get("received_dt") or r.get("receivedDateTime") or "").strip()
        folder = str(r.get("folder") or r.get("folder_path") or "").strip()
        frm = str(r.get("from") or r.get("from_name") or "").strip()
        att = str(r.get("attachment") or "").strip()
        line = f"{i}. {received} [{folder}] {subj}"
        if frm:
            line += f" — From: {frm}"
        if att:
            line += f" — Attachment: {att}"
        if mid:
            line += f"\n   id: {mid}"
        snip = str(r.get("snippet") or r.get("snippet_body") or r.get("snippet_text") or "").strip()
        if snip:
            snip = re.sub(r"\s+", " ", snip)
            if len(snip) > 260:
                snip = snip[:259] + "…"
            line += f"\n   Snippet: {snip}"
        print(line)


def _run_query_strategy(
    con: sqlite3.Connection,
    *,
    strategy: str,
    params: dict,
    limit: int,
) -> tuple[str, list[dict], str]:
    """Return (kind, rows, human_title). kind is 'messages' | 'attachments' | 'text'."""

    limit = max(1, min(50, int(limit)))

    if strategy == "search_messages":
        q = str(params.get("q") or "")
        fts = _fts_query_from_text(q)
        rows = con.execute(
            """
            SELECT
              m.graph_id AS id,
              m.subject AS subject,
              m.received_dt AS received_dt,
              coalesce(m.from_name, m.from_address, '') AS from_name,
              f.folder_path AS folder_path,
              snippet(fts_messages, 1, '', '', ' … ', 12) AS snippet
            FROM fts_messages
            JOIN messages m ON m.id = fts_messages.rowid
            LEFT JOIN folders f ON f.folder_id = m.folder_id
            WHERE fts_messages MATCH ?
            ORDER BY bm25(fts_messages)
            LIMIT ?
            """,
            (fts, limit),
        ).fetchall()
        return ("messages", [dict(r) for r in rows], "Message search")

    if strategy == "messages_on_day":
        day = str(params.get("day") or "").strip()
        sender = str(params.get("sender") or "").strip()
        date_field = str(params.get("date_field") or "received").strip().lower()
        if date_field not in {"received", "sent"}:
            date_field = "received"
        bounds = _day_bounds_utc(day)
        if not bounds:
            return ("text", [{"text": f"Invalid day for messages_on_day: {day!r}"}], "(Invalid day)")
        start_iso, end_iso = bounds

        col = "m.sent_dt" if date_field == "sent" else "m.received_dt"
        where = f"WHERE {col} >= ? AND {col} < ?"
        sql_params: list[Any] = [start_iso, end_iso]

        # If we're filtering by sent date, strongly prefer the Sent Items folder(s).
        if date_field == "sent":
            where += " AND (lower(coalesce(f.folder_path,'')) LIKE '%sent items%' OR lower(coalesce(f.folder_path,'')) = 'sent')"

        if sender:
            where += " AND (lower(coalesce(m.from_name,'')) LIKE ? OR lower(coalesce(m.from_address,'')) LIKE ?)"
            like = f"%{sender.lower()}%"
            sql_params.extend([like, like])

        sql = (
            "SELECT m.graph_id AS id, m.subject AS subject, "
            + col
            + " AS received_dt, "
            + ("'sent'" if date_field == "sent" else "'received'")
            + " AS date_kind, "
            "coalesce(m.from_name, m.from_address, '') AS from_name, f.folder_path AS folder_path "
            "FROM messages m LEFT JOIN folders f ON f.folder_id=m.folder_id "
            + where
            + " ORDER BY "
            + col
            + " DESC LIMIT ?"
        )
        sql_params.append(limit)
        rows = con.execute(sql, sql_params).fetchall()
        label = f"Messages {('sent' if date_field == 'sent' else 'received')} on {day} (UTC)"
        return ("messages", [dict(r) for r in rows], label)

    if strategy == "messages_in_last_days":
        days = params.get("days")
        sender = str(params.get("sender") or "").strip()
        date_field = str(params.get("date_field") or "received").strip().lower()
        if date_field not in {"received", "sent"}:
            date_field = "received"
        try:
            d = max(0, min(365, int(days)))
        except Exception:
            return ("text", [{"text": f"Invalid days for messages_in_last_days: {days!r}"}], "(Invalid days)")

        start_iso = _iso_z(_utc_now() - dt.timedelta(days=d))
        col = "m.sent_dt" if date_field == "sent" else "m.received_dt"
        where = f"WHERE {col} >= ?"
        sql_params: list[Any] = [start_iso]

        if date_field == "sent":
            where += " AND (lower(coalesce(f.folder_path,'')) LIKE '%sent items%' OR lower(coalesce(f.folder_path,'')) = 'sent')"

        if sender:
            where += " AND (lower(coalesce(m.from_name,'')) LIKE ? OR lower(coalesce(m.from_address,'')) LIKE ?)"
            like = f"%{sender.lower()}%"
            sql_params.extend([like, like])

        sql = (
            "SELECT m.graph_id AS id, m.subject AS subject, "
            + col
            + " AS received_dt, "
            + ("'sent'" if date_field == "sent" else "'received'")
            + " AS date_kind, "
            "coalesce(m.from_name, m.from_address, '') AS from_name, f.folder_path AS folder_path "
            "FROM messages m LEFT JOIN folders f ON f.folder_id=m.folder_id "
            + where
            + " ORDER BY "
            + col
            + " DESC LIMIT ?"
        )
        sql_params.append(limit)
        rows = con.execute(sql, sql_params).fetchall()
        return ("messages", [dict(r) for r in rows], f"Messages {('sent' if date_field == 'sent' else 'received')} in the last {d} days")

    if strategy == "search_attachments":
        q = str(params.get("q") or "")
        fts = _fts_query_from_text(q)
        rows = con.execute(
            """
            SELECT
              m.graph_id AS id,
              m.subject AS subject,
              m.received_dt AS received_dt,
              coalesce(m.from_name, m.from_address, '') AS from_name,
              f.folder_path AS folder_path,
              a.name AS attachment,
              a.content_type AS content_type,
              snippet(fts_attachments, 1, '', '', ' … ', 12) AS snippet
            FROM fts_attachments
            JOIN attachments a ON a.id = fts_attachments.rowid
            JOIN messages m ON m.graph_id = a.message_graph_id
            LEFT JOIN folders f ON f.folder_id = m.folder_id
            WHERE fts_attachments MATCH ?
            ORDER BY bm25(fts_attachments)
            LIMIT ?
            """,
            (fts, limit),
        ).fetchall()
        return ("messages", [dict(r) for r in rows], "Attachment content/name search")

    if strategy == "pdf_attachments":
        days = params.get("days")
        sender = str(params.get("sender") or "").strip()
        where = "WHERE (lower(coalesce(a.content_type,''))='application/pdf' OR lower(coalesce(a.name,'')) LIKE '%.pdf')"
        sql_params: list[Any] = []
        if days is not None:
            try:
                d = max(0, min(365, int(days)))
                start_iso = _iso_z(_utc_now() - dt.timedelta(days=d))
                where += " AND m.received_dt >= ?"
                sql_params.append(start_iso)
            except Exception:
                pass
        if sender:
            # Non-parsing: just use the sender string as a LIKE token.
            where += " AND (lower(coalesce(m.from_name,'')) LIKE ? OR lower(coalesce(m.from_address,'')) LIKE ?)"
            like = f"%{sender.lower()}%"
            sql_params.extend([like, like])

        sql = (
            "SELECT DISTINCT m.graph_id AS id, m.subject AS subject, m.received_dt AS received_dt, "
            "coalesce(m.from_name, m.from_address, '') AS from_name, f.folder_path AS folder_path "
            "FROM messages m JOIN attachments a ON a.message_graph_id=m.graph_id "
            "LEFT JOIN folders f ON f.folder_id=m.folder_id "
            + where
            + " ORDER BY m.received_dt DESC LIMIT ?"
        )
        sql_params.append(limit)
        rows = con.execute(sql, sql_params).fetchall()
        return ("messages", [dict(r) for r in rows], "Messages with PDF attachments")

    if strategy == "selected_attachments":
        mid = str(params.get("message_id") or "").strip()
        if not mid:
            return ("text", [], "(No selected message id)")
        rows = con.execute(
            """
            SELECT
              name AS attachment,
              content_type AS content_type,
              size AS size,
              downloaded AS downloaded,
              CASE WHEN content_text IS NOT NULL AND length(content_text) > 0 THEN 1 ELSE 0 END AS has_text,
              substr(coalesce(content_text,''), 1, 600) AS snippet
            FROM attachments
            WHERE message_graph_id = ?
            ORDER BY name ASC
            """,
            (mid,),
        ).fetchall()
        out = [dict(r) for r in rows]
        for r in out:
            r["id"] = mid
        return ("attachments", out, f"Attachments for selected message ({mid})")

    if strategy == "summarize_selected_attachments":
        mid = str(params.get("message_id") or "").strip()
        if not mid:
            return ("text", [], "(No selected message id)")
        rows = con.execute(
            """
            SELECT
              name,
              content_type,
              size,
              downloaded,
              substr(coalesce(content_text,''), 1, 4000) AS content_text
            FROM attachments
            WHERE message_graph_id = ?
            ORDER BY name ASC
            """,
            (mid,),
        ).fetchall()
        atts = [dict(r) for r in rows]
        if not atts:
            return ("text", [], f"(No cached attachments for selected message {mid})")

        key = _openai_key()
        if not key:
            # Deterministic fallback: just list with snippets.
            out_lines = [f"Selected email: {mid}"]
            for i, a in enumerate(atts, 1):
                name = str(a.get("name") or "(unnamed)")
                ct = str(a.get("content_type") or "")
                size = a.get("size")
                downloaded = bool(a.get("downloaded"))
                out_lines.append(f"{i}. {name} ({ct}) {size} bytes — {'downloaded' if downloaded else 'not downloaded'}")
                text = str(a.get("content_text") or "").strip().replace("\n", " ")
                if text:
                    out_lines.append(f"   text: {text[:350]}{'…' if len(text) > 350 else ''}")
            return ("text", [{"text": "\n".join(out_lines)}], "Attachment summary (deterministic)")

        # LLM summary.
        prompt = {
            "selected_message_id": mid,
            "attachments": [
                {
                    "name": a.get("name"),
                    "content_type": a.get("content_type"),
                    "size": a.get("size"),
                    "downloaded": bool(a.get("downloaded")),
                    "extracted_text": (a.get("content_text") or ""),
                }
                for a in atts
            ],
            "instruction": "Summarize the attachments. If extracted_text is empty, say that text was not extracted. Keep it concise.",
        }

        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "gpt-5.1", "input": json.dumps(prompt)},
            timeout=90,
        )
        if r.status_code >= 400:
            return (
                "text",
                [{"text": f"OpenAI error {r.status_code}: {r.text[:400]}"}],
                "Attachment summary (OpenAI error)",
            )
        data = r.json() if r.content else {}
        out_texts: list[str] = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text":
                    out_texts.append(c.get("text", ""))
        return ("text", [{"text": "\n".join(out_texts).strip()}], "Attachment summary (OpenAI)")

    return ("text", [{"text": f"Unknown strategy: {strategy}"}], "(Unknown)")


def _selected_attachments_context(
    con: sqlite3.Connection,
    *,
    message_id: str,
    max_attachments: int = 25,
    snippet_chars: int = 350,
) -> tuple[str, list[dict]]:
    mid = (message_id or "").strip()
    if not mid:
        return ("", [])

    max_attachments = max(1, min(100, int(max_attachments)))
    snippet_chars = max(0, min(4000, int(snippet_chars)))

    try:
        rows = con.execute(
            """
            SELECT
              name AS attachment,
              content_type AS content_type,
              size AS size,
              downloaded AS downloaded,
              CASE WHEN content_text IS NOT NULL AND length(content_text) > 0 THEN 1 ELSE 0 END AS has_text,
              substr(coalesce(content_text,''), 1, ?) AS snippet
            FROM attachments
            WHERE message_graph_id = ?
            ORDER BY name ASC
            LIMIT ?
            """,
            (snippet_chars, mid, max_attachments),
        ).fetchall()
    except Exception:
        return ("", [])

    atts = [dict(r) for r in rows]
    if not atts:
        return ("(No cached attachments for selected email.)", [])

    lines: list[str] = [f"Selected email: {mid}"]
    for i, a in enumerate(atts, 1):
        name = str(a.get("attachment") or "(unnamed)").strip()
        ct = str(a.get("content_type") or "").strip()
        size = a.get("size")
        downloaded = bool(a.get("downloaded"))
        has_text = bool(a.get("has_text"))
        lines.append(
            f"{i}. {name} ({ct}) {size} bytes — "
            + ("downloaded" if downloaded else "not downloaded")
            + ("; text extracted" if has_text else "; no extracted text")
        )
        snip = str(a.get("snippet") or "").strip()
        if snip:
            snip = re.sub(r"\s+", " ", snip)
            if len(snip) > 0:
                lines.append(f"   text: {snip}{'…' if len(snip) >= snippet_chars else ''}")

    # Make the attachment entries look like other results items (id + fields).
    normalized: list[dict] = []
    for a in atts:
        d = dict(a)
        d["id"] = mid
        normalized.append(d)
    return ("\n".join(lines), normalized)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Cache agent CLI (tests natural-language queries against SQLite cache)")
    ap.add_argument(
        "--config",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sync-app", "config.yaml")),
    )
    ap.add_argument("--db", default=None, help="Override SQLite path (else uses storage.sqlite_path from config)")
    ap.add_argument("--model", default=None, help="OpenAI model override (default from config chat.model)")
    ap.add_argument("--no-openai", action="store_true", help="Disable OpenAI planning; use a minimal fallback")
    ap.add_argument(
        "--selected",
        default="__UNSET__",
        help="Selected message graph_id (for attachment summary queries). Use empty string to clear.",
    )
    ap.add_argument("--repl", action="store_true", help="Interactive mode (enter multiple queries)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON only (for web bridge)")
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--state", action="store_true", help="Persist search-agent state into config.yaml")
    mx.add_argument("--no-state", action="store_true", help="Do not persist state into config.yaml")
    ap.add_argument("--limit", default=10, type=int)
    ap.add_argument("--explain", action="store_true", help="Print the chosen strategy + params")
    ap.add_argument("query", nargs="*", help="Query text (if omitted, reads from stdin)")

    ns = ap.parse_args(argv)

    cfg_path = os.path.abspath(ns.config)
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"Config not found: {cfg_path}")
    cfg = _load_yaml_config(cfg_path)

    state_cfg = _agent_state_get(cfg)
    prev_resp_id = str(state_cfg.get("openai_previous_response_id") or "").strip() or None

    db_path = _resolve_sqlite_path(cfg_path, cfg, ns.db)
    db_exists = os.path.exists(db_path)

    if ns.selected != "__UNSET__":
        selected_message_id = (str(ns.selected) if ns.selected is not None else "").strip() or None
    else:
        selected_message_id = None
    if selected_message_id is None and ns.selected == "__UNSET__":
        selected_message_id = str(state_cfg.get("last_selected_message_id") or "").strip() or None

    # Decide whether to persist state.
    persist_state: bool
    if ns.no_state:
        persist_state = False
    elif ns.state:
        persist_state = True
    else:
        # Default: persist when used as a web bridge (json mode), else leave config untouched.
        persist_state = bool(ns.json)

    def run_one(query: str) -> tuple[int, dict]:
        nonlocal selected_message_id
        nonlocal prev_resp_id
        q = (query or "").strip()
        if not q:
            return (0, {"ok": False, "error": "empty query"})

        # Sync/cache status requests are handled deterministically before any LLM planning.
        if _is_sync_status_request(q):
            out = _sync_status(cfg)
            text = (
                f"Sync status: {'RUNNING' if out.get('running') else 'NOT RUNNING'}\n"
                f"pid: {out.get('pid') or ''}\n"
                f"started_at: {out.get('started_at') or ''}\n"
                f"log: {out.get('log_path') or ''}"
            ).strip()
            tail = str(out.get('log_tail') or '').strip()
            if tail:
                text += "\n\nLast log output:\n" + tail.strip()
            payload = {"ok": True, "query": q, "kind": "text", "title": "Sync status", "text": text, **out}
            if persist_state:
                _write_yaml_config(cfg_path, cfg)
            return (0, payload)

        if _is_sync_request(q):
            out = _start_full_sync(cfg_path, cfg)
            if not out.get("ok"):
                payload = {"ok": False, "query": q, "kind": "text", "title": "Sync", "text": str(out.get("error") or "sync failed"), **out}
                return (0, payload)
            text = (
                "Started a full cache sync in the background.\n"
                f"pid: {out.get('pid') or ''}\n"
                f"log: {out.get('log_path') or ''}\n\n"
                "You can ask: \"sync status\" to see progress."
            )
            payload = {"ok": True, "query": q, "kind": "text", "title": "Sync started", "text": text, **out}
            if persist_state:
                _write_yaml_config(cfg_path, cfg)
            return (0, payload)

        if _is_cache_status_request(q):
            if not db_exists:
                payload = {
                    "ok": True,
                    "query": q,
                    "kind": "text",
                    "title": "Cache status",
                    "text": "SQLite cache does not exist yet. You can say 'sync now' to build it.",
                    "action": "cache_status",
                }
                return (0, payload)

            con0 = sqlite3.connect(db_path, timeout=2)
            try:
                s0, e0 = _cache_bounds(con0)
                try:
                    msg_count = int(con0.execute("SELECT count(1) FROM messages").fetchone()[0])
                except Exception:
                    msg_count = 0
                try:
                    att_count = int(con0.execute("SELECT count(1) FROM attachments").fetchone()[0])
                except Exception:
                    att_count = 0
            finally:
                con0.close()

            s_txt = s0.isoformat().replace("+00:00", "Z") if s0 else "(unknown)"
            e_txt = e0.isoformat().replace("+00:00", "Z") if e0 else "(unknown)"
            text = (
                "Cache status:\n"
                f"messages: {msg_count}\n"
                f"attachments: {att_count}\n"
                f"coverage_start: {s_txt}\n"
                f"coverage_end: {e_txt}"
            )
            payload = {
                "ok": True,
                "query": q,
                "kind": "text",
                "title": "Cache status",
                "text": text,
                "action": "cache_status",
                "cache": {"messages": msg_count, "attachments": att_count, "coverage_start": s_txt, "coverage_end": e_txt},
            }
            return (0, payload)

        if not db_exists:
            return (
                0,
                {
                    "ok": False,
                    "error": f"SQLite cache not found: {db_path}. Say 'sync now' to build it.",
                    "query": q,
                },
            )

        # Decide plan.
        plan: dict
        used_openai = False
        if (not ns.no_openai) and _openai_key():
            try:
                plan = _openai_plan(
                    model=_chat_model(cfg, ns.model),
                    message=q,
                    selected_message_id=selected_message_id,
                    previous_response_id=prev_resp_id,
                )
                used_openai = True
            except Exception as e:
                plan = {"strategy": "search_messages", "params": {"q": q}, "limit": ns.limit, "rationale": str(e)}
        else:
            # Minimal fallback: choose a reasonable default without trying to parse.
            lowered = q.lower()
            temporal = _extract_temporal_intent(q)
            if temporal:
                plan = {
                    "strategy": str(temporal.get("strategy")),
                    "params": temporal.get("params") if isinstance(temporal.get("params"), dict) else {},
                    "limit": ns.limit,
                    "rationale": "fallback: temporal intent -> date filter",
                }
            elif "pdf" in lowered and ("attachment" in lowered or "attachments" in lowered or "pdfs" in lowered):
                plan = {
                    "strategy": "pdf_attachments",
                    "params": {
                        "days": _extract_days_hint(q),
                        "sender": _extract_sender_hint(q) or "",
                    },
                    "limit": ns.limit,
                    "rationale": "fallback: pdf attachments query",
                }
            elif "attachment" in lowered or "attachments" in lowered:
                plan = {
                    "strategy": "search_attachments",
                    "params": {"q": q},
                    "limit": ns.limit,
                    "rationale": "fallback",
                }
            else:
                plan = {
                    "strategy": "search_messages",
                    "params": {"q": q},
                    "limit": ns.limit,
                    "rationale": "fallback",
                }

        strategy = str(plan.get("strategy") or "").strip()
        params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
        limit = _safe_int(plan.get("limit"), ns.limit)
        limit = max(1, min(50, int(limit)))

        # Guardrail: temporal intent should be handled via date filters, not literal text search.
        # If we detect temporal intent, prefer the deterministic temporal plan unless the query
        # is explicitly about the selected email's attachments or PDF attachment queries.
        temporal = _extract_temporal_intent(q)
        if temporal and strategy not in {"selected_attachments", "summarize_selected_attachments", "pdf_attachments"}:
            strategy = str(temporal.get("strategy") or strategy)
            params = temporal.get("params") if isinstance(temporal.get("params"), dict) else params

        # Track OpenAI continuity.
        if used_openai:
            rid = str(plan.get("response_id") or "").strip()
            if rid:
                prev_resp_id = rid

        # Normalize planner param aliases (keeps templates simple and safe).
        if isinstance(params, dict) and strategy == "pdf_attachments":
            # Some plans may use from_query/from instead of sender.
            if not params.get("sender"):
                alt = params.get("from_query") or params.get("from")
                if isinstance(alt, str) and alt.strip():
                    params = dict(params)
                    params["sender"] = alt.strip()

        # If the plan wants selected attachments, supply the selected id.
        if strategy in {"selected_attachments", "summarize_selected_attachments"}:
            if "message_id" not in params:
                params = dict(params)
                params["message_id"] = selected_message_id or ""

        effective = {"strategy": strategy, "params": params, "limit": limit}

        if ns.explain and (not ns.json):
            print("== PLAN ==")
            print(json.dumps({"openai_planner": used_openai, **plan}, indent=2, sort_keys=True))
            print("== EFFECTIVE ==")
            print(json.dumps(effective, indent=2, sort_keys=True))

        con = sqlite3.connect(db_path, timeout=2)
        con.row_factory = sqlite3.Row
        selected_context_text = ""
        selected_attachments: list[dict] = []
        notices: list[str] = []
        requested_start: dt.datetime | None = None
        try:
            # For date-based strategies, compute the requested start time for coverage warnings.
            try:
                if str(effective.get("strategy") or "") == "messages_on_day":
                    b = _day_bounds_utc(str((effective.get("params") or {}).get("day") or ""))
                    if b:
                        requested_start = _parse_iso_utc(b[0])
                elif str(effective.get("strategy") or "") == "messages_in_last_days":
                    d_raw = (effective.get("params") or {}).get("days")
                    d_int = int(d_raw) if d_raw is not None else None
                    if d_int is not None:
                        requested_start = _utc_now() - dt.timedelta(days=max(0, min(3650, d_int)))
                elif str(effective.get("strategy") or "") == "pdf_attachments":
                    d_raw = (effective.get("params") or {}).get("days")
                    if d_raw is not None:
                        d_int = int(d_raw)
                        requested_start = _utc_now() - dt.timedelta(days=max(0, min(3650, d_int)))
            except Exception:
                requested_start = None

            kind, rows, title = _run_query_strategy(
                con,
                strategy=str(effective["strategy"]),
                params=effective["params"] if isinstance(effective.get("params"), dict) else {},
                limit=int(effective["limit"]),
            )

            # Coverage warning (best-effort): if user asked for older mail than we have cached.
            if requested_start:
                cov_s, cov_e = _cache_bounds(con)
                if cov_s and requested_start < cov_s:
                    notices.append(
                        "Your query asks for emails older than the current cache coverage. "
                        f"Cache coverage starts at {cov_s.isoformat().replace('+00:00','Z')}. "
                        "You can say 'sync now' to update the cache."
                    )
            if ns.json and selected_message_id:
                selected_context_text, selected_attachments = _selected_attachments_context(
                    con,
                    message_id=selected_message_id,
                    max_attachments=25,
                    snippet_chars=350,
                )
        finally:
            con.close()

        if kind == "text":
            out_text = ""
            if rows and isinstance(rows[0], dict) and "text" in rows[0]:
                out_text = str(rows[0]["text"])
            if not ns.json:
                print("\n==", title, "==")
                print(out_text or "(no text)")

            payload = {
                "ok": True,
                "query": q,
                "selected_message_id": selected_message_id or "",
                "plan": {"openai_planner": used_openai, **plan},
                "effective": effective,
                "kind": "text",
                "title": title,
                "text": out_text,
            }
            if notices:
                payload["notices"] = notices
                payload["notices_text"] = "\n".join([f"NOTICE: {n}" for n in notices])
            if ns.json and selected_message_id:
                payload["selected_context_text"] = selected_context_text
                payload["selected_attachments"] = selected_attachments
            return (0, payload)

        if not ns.json:
            _print_rows(rows, title=title)

        # Normalize results + citations for web UI.
        norm_results: list[dict] = []
        citations: list[dict] = []
        seen: set[str] = set()
        for r in rows:
            d = dict(r)
            mid = str(d.get("id") or d.get("graph_id") or "").strip()
            received_iso = str(
                d.get("received_dt") or d.get("dt_iso") or d.get("received") or d.get("receivedDateTime") or ""
            ).strip()
            date_kind = str(d.get("date_kind") or "received").strip().lower()
            received_human, received_days_ago = _human_received(received_iso)
            if date_kind == "sent" and received_human:
                received_human = "Sent: " + received_human
            folder = str(d.get("folder_path") or d.get("folder") or "").strip()
            subj = str(d.get("subject") or "").strip()
            frm = str(d.get("from_name") or d.get("from") or "").strip()
            att = str(d.get("attachment") or d.get("attachment_name") or "").strip()
            snip = str(d.get("snippet") or d.get("snippet_text") or d.get("snippet_body") or "").strip()
            if snip:
                snip = re.sub(r"\s+", " ", snip)

            norm_results.append(
                {
                    "id": mid,
                    "received": received_human,
                    "received_iso": received_iso,
                    "received_days_ago": received_days_ago,
                    "date_kind": date_kind,
                    "folder": folder,
                    "subject": subj,
                    "from": frm,
                    "attachment": att,
                    "snippet": snip,
                }
            )
            if mid and mid not in seen:
                seen.add(mid)
                citations.append(
                    {
                        "id": mid,
                        "folder": folder,
                        "received": received_human,
                        "received_iso": received_iso,
                        "received_days_ago": received_days_ago,
                        "date_kind": date_kind,
                        "subject": subj,
                        "from": frm,
                    }
                )

        # Text block for downstream LLM answering (stable formatting).
        lines: list[str] = []
        for i, r in enumerate(norm_results, 1):
            lines.append(
                f"{i}. [{r.get('folder','')}] {r.get('received','')} — {r.get('subject','')} — From: {r.get('from','')}\n"
                + (f"   Attachment: {r.get('attachment','')}\n" if r.get("attachment") else "")
                + (f"   Snippet: {r.get('snippet','')}\n" if r.get("snippet") else "")
                + f"   id: {r.get('id','')}"
            )
        context_text = "\n\n".join(lines) if lines else "(No results.)"

        payload = {
            "ok": True,
            "query": q,
            "selected_message_id": selected_message_id or "",
            "plan": {"openai_planner": used_openai, **plan},
            "effective": effective,
            "kind": "messages",
            "title": title,
            "results": norm_results,
            "citations": citations,
            "context_text": context_text,
        }
        if notices:
            payload["notices"] = notices
            payload["notices_text"] = "\n".join([f"NOTICE: {n}" for n in notices])
        if ns.json and selected_message_id:
            payload["selected_context_text"] = selected_context_text
            payload["selected_attachments"] = selected_attachments
        return (0, payload)

    if ns.repl:
        if ns.json:
            raise RuntimeError("--json is not supported with --repl")
        print("Cache agent REPL mode.")
        print("Commands: :help, :selected <message_id>, :exit")
        if selected_message_id:
            print(f"Selected message id: {selected_message_id}")
        while True:
            try:
                line = input("cache-agent> ")
            except EOFError:
                print()
                return 0
            line = (line or "").strip()
            if not line:
                continue
            if line in {":q", ":quit", ":exit"}:
                return 0
            if line in {":h", ":help"}:
                print("\nEnter a natural-language query, e.g.: pdfs from annie last 30 days")
                print("Commands:")
                print("  :selected <message_id>   Set the selected message id")
                print("  :exit                    Quit")
                continue
            if line.lower().startswith(":selected "):
                val = line[len(":selected ") :].strip()
                selected_message_id = val or None
                print(f"Selected message id: {selected_message_id or '(none)'}")
                continue
            run_one(line)
        return 0

    query = " ".join(ns.query).strip()
    if not query:
        query = sys.stdin.read().strip()
    if not query:
        raise RuntimeError("Empty query")
    code, payload = run_one(query)

    # Persist state if requested.
    if persist_state:
        updates = {
            "last_selected_message_id": selected_message_id or "",
            "openai_previous_response_id": prev_resp_id or "",
            "last_query": query,
            "last_run_utc": _iso_z(_utc_now()),
        }
        try:
            _agent_state_set(cfg, updates)
            _write_yaml_config(cfg_path, cfg)
        except Exception:
            # State persistence must never break the agent.
            pass

    if ns.json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.write("\n")
        return 0 if payload.get("ok") else 2

    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
