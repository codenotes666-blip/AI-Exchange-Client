#!/usr/bin/env python3
import datetime as dt
import html
import json
import os
import re
import sqlite3
import base64
import subprocess
import textwrap
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msal
import requests


def _load_yaml_config(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "YAML config present but PyYAML is not installed. "
            "Install via APT: sudo apt install python3-yaml\n"
            f"Import error: {e}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


DEFAULT_CONFIG_PATH = os.environ.get(
    "APP_CONFIG",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sync-app", "config.yaml")),
)

CONFIG_LOCK = threading.Lock()
CONFIG_PATH = os.path.abspath(DEFAULT_CONFIG_PATH)
CFG = _load_yaml_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}

SYNC_LOCK = threading.Lock()
SYNC_PROC: subprocess.Popen | None = None
SYNC_LOG_PATH: str | None = None
SYNC_LOG_FH = None

AUTO_SYNC_STOP = threading.Event()


def _run_search_agent_json(query: str, *, selected_message_id: str | None, limit: int | None = None) -> dict:
    """Run the external search agent and return its JSON output."""
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty query"}

    try:
        lim = int(limit) if limit is not None else int(_chat_cfg_int("chat", "max_results", default=6))
    except Exception:
        lim = int(_chat_cfg_int("chat", "max_results", default=6))
    lim = max(1, min(50, lim))

    cmd = [
        "python3",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agent-app", "cache_agent_cli.py")),
        "--json",
        "--config",
        CONFIG_PATH,
        "--limit",
        str(lim),
    ]
    if selected_message_id is not None:
        cmd.extend(["--selected", str(selected_message_id)])
    cmd.append(q)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_COLOR"] = "1"

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=60)
    except Exception as e:
        return {"ok": False, "error": f"search agent failed to run: {e}"}

    out = (p.stdout or b"").decode("utf-8", errors="replace").strip()
    err = (p.stderr or b"").decode("utf-8", errors="replace").strip()
    if not out:
        return {"ok": False, "error": "search agent returned no output", "stderr": err, "exitCode": p.returncode}

    try:
        j = json.loads(out)
        if isinstance(j, dict):
            if err and "stderr" not in j:
                j["stderr"] = err
            if "exitCode" not in j:
                j["exitCode"] = p.returncode
            return j
        return {"ok": False, "error": "search agent returned non-object JSON", "raw": out[:2000]}
    except Exception:
        return {
            "ok": False,
            "error": "failed to parse search agent JSON",
            "raw": out[:2000],
            "stderr": err,
            "exitCode": p.returncode,
        }


def _reload_config(path: str | None = None):
    global CONFIG_PATH, CFG
    with CONFIG_LOCK:
        if path:
            CONFIG_PATH = os.path.abspath(path)
        CFG = _load_yaml_config(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else {}


def _cfg_get(*keys, default=None):
    cur = CFG
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _resolve_sqlite_path() -> str | None:
    raw = _cfg_get("storage", "sqlite_path", default=None)
    if not raw:
        return None
    p = str(raw)
    cfg_dir = os.path.dirname(os.path.abspath(CONFIG_PATH))
    return os.path.abspath(os.path.join(cfg_dir, p))


def _cache_db_path() -> str | None:
    return _resolve_sqlite_path()


def _sync_attachments_mode() -> str:
    return str(_cfg_get("sync", "attachments", "mode", default="full")).strip().lower()


def _auto_sync_enabled() -> bool:
    try:
        return bool(_cfg_get("sync", "auto_update", "enabled", default=False))
    except Exception:
        return False


def _auto_sync_interval_seconds() -> int:
    try:
        v = int(_cfg_get("sync", "auto_update", "interval_seconds", default=60) or 60)
    except Exception:
        v = 60
    # Keep it sane.
    return max(15, min(3600, v))


def _cache_notify_enabled() -> bool:
    try:
        return bool(_cfg_get("web", "cache_notify", "enabled", default=False))
    except Exception:
        return False


def _cache_notify_interval_seconds() -> int:
    try:
        v = int(_cfg_get("web", "cache_notify", "interval_seconds", default=30) or 30)
    except Exception:
        v = 30
    return max(5, min(3600, v))


def _cache_version() -> float | None:
    """A cheap cache-change indicator based on SQLite file mtime."""
    p = _cache_db_path()
    if not p or not os.path.exists(p):
        return None
    try:
        return float(os.path.getmtime(p))
    except Exception:
        return None


def _db_latest_received_dt(folder_id: str) -> str | None:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return None
    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        row = con.execute(
            "SELECT received_dt FROM messages WHERE folder_id = ? AND received_dt IS NOT NULL ORDER BY received_dt DESC LIMIT 1",
            (folder_id,),
        ).fetchone()
        con.close()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def _check_graph_for_new_messages() -> dict:
    """Check if Graph has newer mail than the local cache (no side effects)."""
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return {"cachePresent": False, "hasNew": False}

    # Avoid overlapping with an active sync.
    with SYNC_LOCK:
        running = SYNC_PROC is not None and SYNC_PROC.poll() is None
    if running:
        return {"running": True, "hasNew": False}

    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        allowed_folder_ids = _chat_selected_folder_ids(con) or []
        con.close()
    except Exception:
        allowed_folder_ids = []

    if not allowed_folder_ids:
        return {"hasNew": False, "reason": "no selected folders"}

    has_new = False
    checked = 0
    for fid in allowed_folder_ids:
        try:
            checked += 1
            cached_latest = _db_latest_received_dt(fid) or ""
            msgs = GRAPH.list_folder_messages(MAILBOX_UPN, fid, top=1) or []
            if not msgs:
                continue
            g_latest = str((msgs[0] or {}).get("receivedDateTime") or "")
            if g_latest and (not cached_latest or g_latest > cached_latest):
                has_new = True
                break
        except Exception:
            continue

    return {"hasNew": bool(has_new), "checkedFolders": checked}


_DB_COL_CACHE: dict[tuple[str, str, str], bool] = {}


def _db_has_column(cache_db_path: str, table: str, column: str) -> bool:
    key = (cache_db_path, table, column)
    if key in _DB_COL_CACHE:
        return _DB_COL_CACHE[key]
    ok = False
    try:
        con = sqlite3.connect(cache_db_path, timeout=1)
        cur = con.execute(f"PRAGMA table_info({table})")
        cols = {str(r[1]) for r in cur.fetchall()}
        con.close()
        ok = column in cols
    except Exception:
        ok = False
    _DB_COL_CACHE[key] = ok
    return ok


def _normalize_content_id(value: str | None) -> str:
    s = (value or "").strip()
    if s.lower().startswith("cid:"):
        s = s[4:]
    if s.startswith("<") and s.endswith(">") and len(s) >= 2:
        s = s[1:-1]
    return s.strip().lower()


_CID_SRC_RE = re.compile(r"(?i)(src)=(['\"])\s*cid:([^'\"\s>]+)\2")
_CID_URL_RE = re.compile(r"(?i)url\(\s*cid:([^\)\s]+)\s*\)")


def _rewrite_cid_images(body_html: str, message_id: str) -> str:
    if not body_html:
        return body_html

    def repl_src(m):
        attr = m.group(1)
        q = m.group(2)
        cid = m.group(3)
        href = "/inline?mid=" + urllib.parse.quote(message_id, safe="") + "&cid=" + urllib.parse.quote(cid, safe="")
        return f"{attr}={q}{href}{q}"

    def repl_url(m):
        cid = m.group(1)
        href = "/inline?mid=" + urllib.parse.quote(message_id, safe="") + "&cid=" + urllib.parse.quote(cid, safe="")
        return f"url({href})"

    s = _CID_SRC_RE.sub(repl_src, body_html)
    s = _CID_URL_RE.sub(repl_url, s)
    return s


def _db_list_attachments(message_graph_id: str) -> list[dict] | None:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return None
    try:
        con = sqlite3.connect(cache_db_path, timeout=1)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            """
            SELECT
              attachment_id,
              name,
              content_type,
              size,
              is_inline,
              downloaded,
              CASE WHEN content_bytes IS NULL THEN 0 ELSE 1 END AS has_bytes
            FROM attachments
            WHERE message_graph_id = ?
            ORDER BY name ASC
            """,
            (message_graph_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return None


def _db_get_attachment_bytes(message_graph_id: str, attachment_id: str) -> tuple[bytes | None, dict | None]:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return (None, None)
    try:
        con = sqlite3.connect(cache_db_path, timeout=1)
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT attachment_id, name, content_type, size, content_bytes, downloaded, error
            FROM attachments
            WHERE message_graph_id = ? AND attachment_id = ?
            """,
            (message_graph_id, attachment_id),
        ).fetchone()
        con.close()
        if not row:
            return (None, None)
        meta = dict(row)
        b = meta.pop("content_bytes", None)
        return (b, meta)
    except Exception:
        return (None, None)


def _db_upsert_attachment_bytes(
    message_graph_id: str,
    attachment_id: str,
    *,
    name: str | None,
    content_type: str | None,
    size: int | None,
    content_bytes: bytes | None,
    downloaded: int,
    error: str | None,
):
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return
    try:
        con = sqlite3.connect(cache_db_path, timeout=5)
        con.execute(
            """
            INSERT INTO attachments(
              message_graph_id, attachment_id, name, content_type, size, content_bytes, downloaded, error
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(message_graph_id, attachment_id) DO UPDATE SET
              name=coalesce(excluded.name, attachments.name),
              content_type=coalesce(excluded.content_type, attachments.content_type),
              size=coalesce(excluded.size, attachments.size),
              content_bytes=excluded.content_bytes,
              downloaded=excluded.downloaded,
              error=excluded.error
            """,
            (
                message_graph_id,
                attachment_id,
                name,
                content_type,
                size,
                content_bytes,
                int(downloaded),
                error,
            ),
        )
        con.commit()
        con.close()
    except Exception:
        return


def _db_list_messages(folder_id: str, top: int = 10) -> list[dict] | None:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return None
    try:
        con = sqlite3.connect(cache_db_path, timeout=1)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            """
            SELECT
              graph_id AS id,
              subject,
                            from_name,
              from_address,
                            to_json,
              received_dt AS receivedDateTime,
              body_preview AS bodyPreview,
              has_attachments AS hasAttachments
            FROM messages
            WHERE folder_id = ?
            ORDER BY received_dt DESC
            LIMIT ?
            """,
            (folder_id, int(top)),
        )
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        for r in rows:
            r["hasAttachments"] = bool(r.get("hasAttachments"))
        return rows
    except Exception:
        return None


def _db_get_message(message_graph_id: str) -> dict | None:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return None
    try:
        has_html = _db_has_column(cache_db_path, "messages", "body_html")
        con = sqlite3.connect(cache_db_path, timeout=1)
        con.row_factory = sqlite3.Row
        if has_html:
            row = con.execute(
                """
                SELECT
                  graph_id AS id,
                  subject,
                                    from_name,
                  from_address,
                                    to_json,
                                    cc_json,
                  received_dt AS receivedDateTime,
                  body_text AS bodyText,
                  body_html AS bodyHtml,
                  has_attachments AS hasAttachments
                FROM messages
                WHERE graph_id = ?
                """,
                (message_graph_id,),
            ).fetchone()
        else:
            row = con.execute(
                """
                SELECT
                  graph_id AS id,
                  subject,
                                    from_name,
                  from_address,
                                    to_json,
                                    cc_json,
                  received_dt AS receivedDateTime,
                  body_text AS bodyText,
                  has_attachments AS hasAttachments
                FROM messages
                WHERE graph_id = ?
                """,
                (message_graph_id,),
            ).fetchone()
        con.close()
        if not row:
            return None
        d = dict(row)
        d["hasAttachments"] = bool(d.get("hasAttachments"))
        return d
    except Exception:
        return None


_CHAT_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "about",
    "do",
    "does",
    "did",
    "email",
    "emails",
    "from",
    "have",
    "i",
    "in",
    "is",
    "latest",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "sent",
    "the",
    "their",
    "to",
    "today",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
}


def _chat_extract_query_parts(message: str) -> tuple[str, list[str]]:
    """Extract a focused retrieval query from a natural-language chat message.

    Returns (mode, tokens) where mode is one of: 'any', 'from', 'to'.
    """
    s = (message or "").strip()
    if not s:
        return ("any", [])

    lowered = s.lower()

    mode = "any"
    # Prefer explicit intent like "from Annie" / "to Annie".
    m_from = re.search(r"\bfrom\b\s+(.+)$", lowered)
    m_to = re.search(r"\bto\b\s+(.+)$", lowered)
    if m_from:
        mode = "from"
        focus = s[m_from.start(1) :].strip()
    elif m_to:
        mode = "to"
        focus = s[m_to.start(1) :].strip()
    else:
        focus = s

    # Tokenize on common email/name characters.
    raw_tokens = re.findall(r"[A-Za-z0-9@._+'\-]+", focus)
    tokens: list[str] = []
    for t in raw_tokens:
        t2 = t.strip().strip("'\"")
        if not t2:
            continue
        tl = t2.lower()
        if tl in _CHAT_STOPWORDS:
            continue
        # Ignore 1-2 char noise tokens unless they look like an email fragment.
        if len(t2) < 3 and "@" not in t2:
            continue
        tokens.append(t2)

    # Keep it small and high-signal.
    return (mode, tokens[:6])


def _chat_build_fts_query(tokens: list[str]) -> str:
    """Build an FTS5 MATCH string that is tolerant of names and prefixes."""
    if not tokens:
        return ""
    parts = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        # Escape quotes for safety.
        t = t.replace('"', "")
        # Prefix-match for simple alpha tokens (names).
        if re.fullmatch(r"[A-Za-z]{3,}", t):
            parts.append(t + "*")
        else:
            parts.append('"' + t + '"')
    # OR tends to work better for short "name" queries.
    return " OR ".join(parts)


def _chat_cfg_bool(*keys, default=False) -> bool:
    try:
        return bool(_cfg_get(*keys, default=default))
    except Exception:
        return bool(default)


def _chat_cfg_int(*keys, default: int) -> int:
    try:
        return int(_cfg_get(*keys, default=default) or default)
    except Exception:
        return int(default)


def _chat_cfg_str(*keys, default: str) -> str:
    try:
        v = _cfg_get(*keys, default=default)
        return str(v) if v is not None else str(default)
    except Exception:
        return str(default)


def _chat_selected_folder_ids(con: sqlite3.Connection) -> list[str] | None:
    # Restrict retrieval to configured folders (sync.folders.names), if possible.
    try:
        selected = _cfg_get("sync", "folders", "names", default=None)
        if not isinstance(selected, list) or not selected:
            return None
        selected_norm = {str(x).strip().lower() for x in selected if str(x).strip()}
        if not selected_norm:
            return None

        rows = con.execute("SELECT folder_id, folder_path FROM folders").fetchall()
        out = []
        for r in rows:
            fid = str(r[0] or "")
            p = str(r[1] or "")
            if not fid:
                continue
            p_norm = p.strip().lower()
            leaf = p_norm.split(" / ")[-1] if p_norm else ""
            if p_norm in selected_norm or leaf in selected_norm:
                out.append(fid)
        return out or None
    except Exception:
        return None


def _db_chat_search(query: str) -> tuple[str, list[dict]]:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return ("(Cache DB not found; run sync.)", [])

    enabled = _chat_cfg_bool("chat", "enabled", default=True)
    if not enabled:
        return ("(Chat retrieval disabled.)", [])

    max_results = max(1, min(25, _chat_cfg_int("chat", "max_results", default=6)))
    max_snip = max(80, min(2000, _chat_cfg_int("chat", "max_snippet_chars", default=400)))
    max_ctx = max(500, min(20000, _chat_cfg_int("chat", "max_context_chars", default=4000)))

    message = (query or "").strip().replace("\n", " ")
    if not message:
        return ("(Empty query.)", [])

    mode, tokens = _chat_extract_query_parts(message)
    focus_text = " ".join(tokens).strip() or message
    fts_query = _chat_build_fts_query(tokens) or focus_text

    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        con.row_factory = sqlite3.Row

        allowed_folder_ids = _chat_selected_folder_ids(con)

        # Prefer FTS if present.
        fts_ok = False
        try:
            row = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_messages'"
            ).fetchone()
            fts_ok = bool(row)
        except Exception:
            fts_ok = False

        results: list[dict] = []

        if fts_ok:
            where_extra = ""
            params: list = [fts_query]
            if allowed_folder_ids:
                where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                params.extend(allowed_folder_ids)
            params.append(int(max_results))
            try:
                rows = con.execute(
                    """
                    SELECT
                      m.graph_id AS graph_id,
                      m.subject AS subject,
                      m.received_dt AS received_dt,
                      m.from_name AS from_name,
                      m.from_address AS from_address,
                      f.folder_path AS folder_path,
                      snippet(fts_messages, 1, '', '', ' … ', 12) AS snippet_body,
                      bm25(fts_messages) AS score
                    FROM fts_messages
                    JOIN messages m ON m.id = fts_messages.rowid
                    LEFT JOIN folders f ON f.folder_id = m.folder_id
                    WHERE fts_messages MATCH ?
                    """
                    + where_extra
                    + " ORDER BY score LIMIT ?",
                    params,
                ).fetchall()
                results = [dict(r) for r in rows]
            except Exception:
                results = []

        # Fallback: LIKE search.
        if not results:
            like = f"%{focus_text}%"
            where_extra = ""
            from_only = mode == "from"

            from_where = "(m.from_name LIKE ? OR m.from_address LIKE ?)"
            any_where = "(m.subject LIKE ? OR m.body_text LIKE ? OR m.from_name LIKE ? OR m.from_address LIKE ?)"
            where_clause = from_where if from_only else any_where
            params2: list = [like, like] if from_only else [like, like, like, like]

            if allowed_folder_ids:
                where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                params2.extend(allowed_folder_ids)
            params2.append(int(max_results))

            rows = con.execute(
                """
                SELECT
                  m.graph_id AS graph_id,
                  m.subject AS subject,
                  m.received_dt AS received_dt,
                  m.from_name AS from_name,
                  m.from_address AS from_address,
                  f.folder_path AS folder_path,
                  substr(coalesce(m.body_text,''), 1, 600) AS snippet_body
                FROM messages m
                LEFT JOIN folders f ON f.folder_id = m.folder_id
                WHERE """
                + where_clause
                + where_extra
                + " ORDER BY m.received_dt DESC LIMIT ?",
                params2,
            ).fetchall()
            results = [dict(r) for r in rows]

        con.close()

        if not results:
            return ("(No matching cached emails found.)", [])

        lines: list[str] = []
        used = 0
        cites: list[dict] = []
        seen_msg_ids: set[str] = set()
        for i, r in enumerate(results, 1):
            subj = (r.get("subject") or "(no subject)").strip()
            received = (r.get("received_dt") or "")
            folder = (r.get("folder_path") or "").strip()
            frm_name = (r.get("from_name") or "").strip()
            frm_addr = (r.get("from_address") or "").strip()
            frm = (frm_name or frm_addr or "").strip()
            snip = (r.get("snippet_body") or "").strip().replace("\n", " ")
            if len(snip) > max_snip:
                snip = snip[: max_snip - 1].rstrip() + "…"
            gid = (r.get("graph_id") or "").strip()

            entry = (
                f"{i}. [{folder}] {received} — {subj} — From: {frm}\n"
                f"   Snippet: {snip}\n"
                f"   id: {gid}"
            )
            if used + len(entry) + 2 > max_ctx:
                break
            used += len(entry) + 2
            lines.append(entry)
            cites.append(
                {
                    "id": gid,
                    "folder": folder,
                    "received": received,
                    "subject": subj,
                    "from": frm,
                }
            )

        return ("\n\n".join(lines), cites)
    except Exception:
        return ("(Cache search failed.)", [])


def _db_chat_search_attachments(query: str) -> tuple[str, list[dict]]:
    """Search cached attachment names and extracted text (fts_attachments)."""
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return ("(Cache DB not found; run sync.)", [])

    enabled = _chat_cfg_bool("chat", "enabled", default=True)
    if not enabled:
        return ("(Chat retrieval disabled.)", [])

    max_results = max(1, min(25, _chat_cfg_int("chat", "max_results", default=6)))
    max_snip = max(80, min(2000, _chat_cfg_int("chat", "max_snippet_chars", default=400)))
    max_ctx = max(500, min(20000, _chat_cfg_int("chat", "max_context_chars", default=4000)))

    message = (query or "").strip().replace("\n", " ")
    if not message:
        return ("(Empty query.)", [])

    mode, tokens = _chat_extract_query_parts(message)
    focus_text = " ".join(tokens).strip() or message
    fts_query = _chat_build_fts_query(tokens) or focus_text

    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        con.row_factory = sqlite3.Row
        allowed_folder_ids = _chat_selected_folder_ids(con)

        fts_ok = False
        try:
            row = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_attachments'"
            ).fetchone()
            fts_ok = bool(row)
        except Exception:
            fts_ok = False

        results: list[dict] = []
        if fts_ok:
            where_extra = ""
            params: list = [fts_query]
            if allowed_folder_ids:
                where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                params.extend(allowed_folder_ids)
            params.append(int(max_results))
            try:
                rows = con.execute(
                    """
                    SELECT
                      m.graph_id AS graph_id,
                      m.subject AS subject,
                      m.received_dt AS received_dt,
                      m.from_name AS from_name,
                      m.from_address AS from_address,
                      f.folder_path AS folder_path,
                      a.name AS attachment_name,
                      a.content_type AS content_type,
                      snippet(fts_attachments, 1, '', '', ' … ', 12) AS snippet_text,
                      bm25(fts_attachments) AS score
                    FROM fts_attachments
                    JOIN attachments a ON a.id = fts_attachments.rowid
                    JOIN messages m ON m.graph_id = a.message_graph_id
                    LEFT JOIN folders f ON f.folder_id = m.folder_id
                    WHERE fts_attachments MATCH ?
                    """
                    + where_extra
                    + " ORDER BY score LIMIT ?",
                    params,
                ).fetchall()
                results = [dict(r) for r in rows]
            except Exception:
                results = []

        if not results:
            like = f"%{focus_text}%"
            where_extra = ""
            params2: list = [like, like]
            if allowed_folder_ids:
                where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                params2.extend(allowed_folder_ids)
            params2.append(int(max_results))
            rows = con.execute(
                """
                SELECT
                  m.graph_id AS graph_id,
                  m.subject AS subject,
                  m.received_dt AS received_dt,
                  m.from_name AS from_name,
                  m.from_address AS from_address,
                  f.folder_path AS folder_path,
                  a.name AS attachment_name,
                  a.content_type AS content_type,
                  substr(coalesce(a.content_text,''), 1, 600) AS snippet_text
                FROM attachments a
                JOIN messages m ON m.graph_id = a.message_graph_id
                LEFT JOIN folders f ON f.folder_id = m.folder_id
                WHERE (a.name LIKE ? OR a.content_text LIKE ?)
                """
                + where_extra
                + " ORDER BY m.received_dt DESC LIMIT ?",
                params2,
            ).fetchall()
            results = [dict(r) for r in rows]

        con.close()

        if not results:
            return ("(No matching cached attachments found.)", [])

        lines: list[str] = []
        used = 0
        cites: list[dict] = []
        for i, r in enumerate(results, 1):
            subj = (r.get("subject") or "(no subject)").strip()
            received = (r.get("received_dt") or "")
            folder = (r.get("folder_path") or "").strip()
            frm_name = (r.get("from_name") or "").strip()
            frm_addr = (r.get("from_address") or "").strip()
            frm = (frm_name or frm_addr or "").strip()
            att_name = (r.get("attachment_name") or "(unnamed attachment)").strip()
            ct = (r.get("content_type") or "")
            snip = (r.get("snippet_text") or "").strip().replace("\n", " ")
            if len(snip) > max_snip:
                snip = snip[: max_snip - 1].rstrip() + "…"
            gid = (r.get("graph_id") or "").strip()

            entry = (
                f"{i}. [{folder}] {received} — {subj} — From: {frm}\n"
                f"   Attachment: {att_name} ({ct})\n"
                f"   Snippet: {snip}\n"
                f"   id: {gid}"
            )
            if used + len(entry) + 2 > max_ctx:
                break
            used += len(entry) + 2
            lines.append(entry)
            if gid and gid not in seen_msg_ids:
                seen_msg_ids.add(gid)
                cites.append(
                    {
                        "id": gid,
                        "folder": folder,
                        "received": received,
                        "subject": subj,
                        "from": frm,
                        "attachment": att_name,
                        "content_type": ct,
                    }
                )

        return ("\n\n".join(lines), cites)
    except Exception:
        return ("(Attachment cache search failed.)", [])


def _chat_parse_days_hint(message: str) -> int | None:
    s = (message or "").lower()
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
    return None


def _db_messages_with_pdf_attachments(
    *,
    days: int | None,
    sender_tokens: list[str] | None,
    limit: int,
) -> tuple[str, list[dict]]:
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return ("(Cache DB not found; run sync.)", [])

    limit = max(1, min(50, int(limit)))
    start_iso = None
    if days is not None:
        days = max(0, min(365, int(days)))
        start_dt = _utc_now() - dt.timedelta(days=days)
        start_iso = _iso_z(start_dt)

    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        con.row_factory = sqlite3.Row
        allowed_folder_ids = _chat_selected_folder_ids(con)

        where_extra = ""
        params: list = []
        if start_iso:
            where_extra += " AND m.received_dt >= ?"
            params.append(start_iso)

        # If the user asked for "pdfs from Annie", apply sender filtering.
        sender_tokens = [str(t).strip() for t in (sender_tokens or []) if str(t).strip()]
        if sender_tokens:
            for tok in sender_tokens[:4]:
                like = f"%{tok.lower()}%"
                where_extra += " AND (lower(coalesce(m.from_name,'')) LIKE ? OR lower(coalesce(m.from_address,'')) LIKE ?)"
                params.extend([like, like])

        if allowed_folder_ids:
            where_extra += " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
            params.extend(allowed_folder_ids)
        params.append(int(limit))

        rows = con.execute(
            """
            SELECT DISTINCT
              m.graph_id AS graph_id,
              m.subject AS subject,
              m.received_dt AS received_dt,
              m.from_name AS from_name,
              m.from_address AS from_address,
              f.folder_path AS folder_path
            FROM messages m
            JOIN attachments a ON a.message_graph_id = m.graph_id
            LEFT JOIN folders f ON f.folder_id = m.folder_id
            WHERE (
                lower(coalesce(a.content_type,'')) = 'application/pdf'
                OR lower(coalesce(a.name,'')) LIKE '%.pdf'
              )
            """
            + where_extra
            + " ORDER BY m.received_dt DESC LIMIT ?",
            params,
        ).fetchall()
        con.close()

        results = [dict(r) for r in rows]
        if not results:
            return ("(No matching cached emails with PDF attachments found.)", [])

        lines: list[str] = []
        cites: list[dict] = []
        for i, r in enumerate(results, 1):
            subj = (r.get("subject") or "(no subject)").strip()
            received = (r.get("received_dt") or "")
            folder = (r.get("folder_path") or "").strip()
            frm_name = (r.get("from_name") or "").strip()
            frm_addr = (r.get("from_address") or "").strip()
            frm = (frm_name or frm_addr or "").strip()
            gid = (r.get("graph_id") or "").strip()
            lines.append(f"{i}. [{folder}] {received} — {subj} — From: {frm}\n   id: {gid}")
            cites.append({"id": gid, "folder": folder, "received": received, "subject": subj, "from": frm})

        return ("\n\n".join(lines), cites)
    except Exception:
        return ("(PDF attachment search failed.)", [])


def _db_attachment_summary_for_message(message_graph_id: str) -> str:
    """Return a compact attachment summary for a specific message from cache."""
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return "(Cache DB not found.)"
    mid = (message_graph_id or "").strip()
    if not mid:
        return "(No selected message.)"
    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
              attachment_id,
              name,
              content_type,
              size,
              is_inline,
              downloaded,
              CASE WHEN content_bytes IS NULL THEN 0 ELSE 1 END AS has_bytes,
              substr(coalesce(content_text,''), 1, 1200) AS content_text
            FROM attachments
            WHERE message_graph_id = ?
            ORDER BY name ASC
            """,
            (mid,),
        ).fetchall()
        con.close()
        if not rows:
            return "(No cached attachments for the selected email.)"

        lines = ["Selected email attachments (cached):"]
        for i, r in enumerate(rows, 1):
            name = str(r["name"] or "(unnamed)")
            ct = str(r["content_type"] or "")
            size = r["size"]
            inline = bool(r["is_inline"]) if r["is_inline"] is not None else False
            downloaded = bool(r["downloaded"]) if r["downloaded"] is not None else False
            has_bytes = bool(r["has_bytes"]) if r["has_bytes"] is not None else False
            text_snip = (r["content_text"] or "").strip().replace("\n", " ")
            if len(text_snip) > 350:
                text_snip = text_snip[:349].rstrip() + "…"

            meta_bits = []
            if ct:
                meta_bits.append(ct)
            if isinstance(size, int) and size >= 0:
                meta_bits.append(f"{size} bytes")
            if inline:
                meta_bits.append("inline")
            meta_bits.append("downloaded" if downloaded else "not downloaded")
            meta_bits.append("has bytes" if has_bytes else "no bytes")
            meta = ", ".join(meta_bits)

            lines.append(f"{i}. {name} ({meta})")
            if text_snip:
                lines.append(f"   Extracted text: {text_snip}")

        return "\n".join(lines)
    except Exception:
        return "(Failed to read cached attachments.)"


def _db_search_messages_for_ui(search_query: str) -> list[dict] | None:
    """Return message summaries to populate the middle pane for a search query."""
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return None

    q_raw = (search_query or "").strip().replace("\n", " ")
    if not q_raw:
        return None

    # Reuse chat config defaults for the amount of content shown.
    max_results = max(1, min(50, _chat_cfg_int("chat", "max_results", default=6)))

    mode, tokens = _chat_extract_query_parts(q_raw)
    focus_text = " ".join(tokens).strip() or q_raw
    fts_query = _chat_build_fts_query(tokens) or focus_text

    lowered = q_raw.lower()
    wants_attachments = any(
        k in lowered
        for k in (
            "attachment",
            "attachments",
            "pdf",
            "docx",
            "xlsx",
            "ppt",
            "spreadsheet",
            "file",
        )
    )

    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        con.row_factory = sqlite3.Row

        allowed_folder_ids = _chat_selected_folder_ids(con)

        fts_ok = False
        try:
            row = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_messages'"
            ).fetchone()
            fts_ok = bool(row)
        except Exception:
            fts_ok = False

        rows = []
        if fts_ok:
            where_extra = ""
            params: list = [fts_query]
            if allowed_folder_ids:
                where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                params.extend(allowed_folder_ids)
            params.append(int(max_results))
            try:
                rows = con.execute(
                    """
                    SELECT
                      m.graph_id AS id,
                      m.folder_id AS folder_id,
                      f.folder_path AS folder_path,
                      m.subject AS subject,
                      m.from_name AS from_name,
                      m.from_address AS from_address,
                      m.to_json AS to_json,
                      m.received_dt AS receivedDateTime,
                      m.has_attachments AS hasAttachments,
                      substr(coalesce(m.body_text,''), 1, 600) AS bodyPreview,
                      bm25(fts_messages) AS score
                    FROM fts_messages
                    JOIN messages m ON m.id = fts_messages.rowid
                    LEFT JOIN folders f ON f.folder_id = m.folder_id
                    WHERE fts_messages MATCH ?
                    """
                    + where_extra
                    + " ORDER BY score LIMIT ?",
                    params,
                ).fetchall()
            except Exception:
                rows = []

        if not rows:
            like = f"%{focus_text}%"
            where_extra = ""
            from_only = mode == "from"
            from_where = "(m.from_name LIKE ? OR m.from_address LIKE ?)"
            any_where = "(m.subject LIKE ? OR m.body_text LIKE ? OR m.from_name LIKE ? OR m.from_address LIKE ?)"
            where_clause = from_where if from_only else any_where
            params2: list = [like, like] if from_only else [like, like, like, like]
            if allowed_folder_ids:
                where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                params2.extend(allowed_folder_ids)
            params2.append(int(max_results))
            rows = con.execute(
                """
                SELECT
                  m.graph_id AS id,
                  m.folder_id AS folder_id,
                  f.folder_path AS folder_path,
                  m.subject AS subject,
                  m.from_name AS from_name,
                  m.from_address AS from_address,
                  m.to_json AS to_json,
                  m.received_dt AS receivedDateTime,
                  m.has_attachments AS hasAttachments,
                  substr(coalesce(m.body_text,''), 1, 600) AS bodyPreview
                FROM messages m
                LEFT JOIN folders f ON f.folder_id = m.folder_id
                WHERE """
                + where_clause
                + where_extra
                + " ORDER BY m.received_dt DESC LIMIT ?",
                params2,
            ).fetchall()

        # If the query looks attachment-oriented, try attachment FTS too.
        # This keeps the search list stable across reloads after chat populates ?search=.
        if (not rows) and wants_attachments:
            # Special-case: "last N days" + PDF attachment queries.
            if "pdf" in lowered:
                days_hint = _chat_parse_days_hint(q_raw)
            else:
                days_hint = None

            if ("pdf" in lowered) and (mode == "from"):
                # Keep behavior aligned with chat routing for the golden query.
                ctx_text, cites = _db_messages_with_pdf_attachments(
                    days=days_hint,
                    sender_tokens=tokens,
                    limit=int(max_results),
                )
                ids = [str(c.get("id") or "") for c in (cites or []) if str(c.get("id") or "").strip()]
                if ids:
                    where_extra = ""
                    params4: list = []
                    if allowed_folder_ids:
                        where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                        params4.extend(allowed_folder_ids)
                    params4.extend(ids)
                    rows = con.execute(
                        """
                        SELECT
                          m.graph_id AS id,
                          m.folder_id AS folder_id,
                          f.folder_path AS folder_path,
                          m.subject AS subject,
                          m.from_name AS from_name,
                          m.from_address AS from_address,
                          m.to_json AS to_json,
                          m.received_dt AS receivedDateTime,
                          m.has_attachments AS hasAttachments,
                          substr(coalesce(m.body_text,''), 1, 600) AS bodyPreview
                        FROM messages m
                        LEFT JOIN folders f ON f.folder_id = m.folder_id
                        WHERE 1=1
                        """
                        + where_extra
                        + " AND m.graph_id IN ("
                        + ",".join(["?"] * len(ids))
                        + ") ORDER BY m.received_dt DESC",
                        params4,
                    ).fetchall()
            elif days_hint is not None:
                start_dt = _utc_now() - dt.timedelta(days=int(days_hint))
                start_iso = _iso_z(start_dt)
                where_extra = ""
                params3: list = [start_iso]
                if allowed_folder_ids:
                    where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                    params3.extend(allowed_folder_ids)
                params3.append(int(max_results))
                rows = con.execute(
                    """
                    SELECT DISTINCT
                      m.graph_id AS id,
                      m.folder_id AS folder_id,
                      f.folder_path AS folder_path,
                      m.subject AS subject,
                      m.from_name AS from_name,
                      m.from_address AS from_address,
                      m.to_json AS to_json,
                      m.received_dt AS receivedDateTime,
                      m.has_attachments AS hasAttachments,
                      substr(coalesce(m.body_text,''), 1, 600) AS bodyPreview
                    FROM messages m
                    JOIN attachments a ON a.message_graph_id = m.graph_id
                    LEFT JOIN folders f ON f.folder_id = m.folder_id
                    WHERE m.received_dt >= ?
                      AND (
                        lower(coalesce(a.content_type,'')) = 'application/pdf'
                        OR lower(coalesce(a.name,'')) LIKE '%.pdf'
                      )
                    """
                    + where_extra
                    + " ORDER BY m.received_dt DESC LIMIT ?",
                    params3,
                ).fetchall()
            else:
                fts_att_ok = False
                try:
                    row = con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_attachments'"
                    ).fetchone()
                    fts_att_ok = bool(row)
                except Exception:
                    fts_att_ok = False

                if fts_att_ok:
                    where_extra = ""
                    params3: list = [fts_query]
                    if allowed_folder_ids:
                        where_extra = " AND m.folder_id IN (" + ",".join(["?"] * len(allowed_folder_ids)) + ")"
                        params3.extend(allowed_folder_ids)
                    params3.append(int(max_results))
                    try:
                        rows = con.execute(
                            """
                            SELECT DISTINCT
                              m.graph_id AS id,
                              m.folder_id AS folder_id,
                              f.folder_path AS folder_path,
                              m.subject AS subject,
                              m.from_name AS from_name,
                              m.from_address AS from_address,
                              m.to_json AS to_json,
                              m.received_dt AS receivedDateTime,
                              m.has_attachments AS hasAttachments,
                              substr(coalesce(m.body_text,''), 1, 600) AS bodyPreview,
                              bm25(fts_attachments) AS score
                            FROM fts_attachments
                            JOIN attachments a ON a.id = fts_attachments.rowid
                            JOIN messages m ON m.graph_id = a.message_graph_id
                            LEFT JOIN folders f ON f.folder_id = m.folder_id
                            WHERE fts_attachments MATCH ?
                            """
                            + where_extra
                            + " ORDER BY score LIMIT ?",
                            params3,
                        ).fetchall()
                    except Exception:
                        rows = []

        con.close()
        out = [dict(r) for r in rows]
        for r in out:
            r["hasAttachments"] = bool(r.get("hasAttachments"))
        return out
    except Exception:
        return None


TENANT_ID_ENV = str(_cfg_get("graph", "tenant_id_env", default="TENANT_ID"))
CLIENT_ID_ENV = str(_cfg_get("graph", "client_id_env", default="CLIENT_ID"))
CLIENT_SECRET_ENV = str(_cfg_get("graph", "client_secret_env", default="CLIENT_SECRET"))
MAILBOX_UPN_ENV = str(_cfg_get("mailbox", "mailbox_upn_env", default="MAILBOX_UPN"))


TENANT_ID = os.environ.get(TENANT_ID_ENV)
CLIENT_ID = os.environ.get(CLIENT_ID_ENV)
CLIENT_SECRET = os.environ.get(CLIENT_SECRET_ENV)
MAILBOX_UPN = os.environ.get(MAILBOX_UPN_ENV)

REQUIRED = {
    TENANT_ID_ENV: TENANT_ID,
    CLIENT_ID_ENV: CLIENT_ID,
    CLIENT_SECRET_ENV: CLIENT_SECRET,
    MAILBOX_UPN_ENV: MAILBOX_UPN,
}

missing = [k for k, v in REQUIRED.items() if not v]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _quote_path(segment: str) -> str:
    # Encode for a single URL path segment (Graph IDs can contain characters like '/' that must be escaped).
    return urllib.parse.quote(segment, safe="")


class GraphClient:
    def __init__(self):
        self._app = msal.ConfidentialClientApplication(
            CLIENT_ID,
            authority=AUTHORITY,
            client_credential=CLIENT_SECRET,
        )
        self._access_token = None
        self._access_token_expires_at = 0
        self._folders_cache = None
        self._folders_cache_expires_at = 0

    def _get_token(self) -> str:
        now = int(time.time())
        if self._access_token and now < (self._access_token_expires_at - 60):
            return self._access_token

        result = self._app.acquire_token_for_client(scopes=SCOPE)
        if "access_token" not in result:
            raise RuntimeError(f"Graph auth failed: {result}")

        self._access_token = result["access_token"]
        expires_in = int(result.get("expires_in", 3599))
        self._access_token_expires_at = now + expires_in
        return self._access_token

    def _headers(self, extra=None):
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        if extra:
            headers.update(extra)
        return headers

    def get_json(self, url: str, *, params=None, headers=None, timeout=30):
        r = requests.get(url, headers=self._headers(headers), params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _paged_values(self, url: str, *, params=None, headers=None, timeout=30):
        values = []
        next_url = url
        next_params = params
        while next_url:
            data = self.get_json(next_url, params=next_params, headers=headers, timeout=timeout)
            values.extend(data.get("value", []))
            next_url = data.get("@odata.nextLink")
            next_params = None
        return values

    def list_folders(self, upn: str):
        now = int(time.time())
        if self._folders_cache and now < self._folders_cache_expires_at:
            return self._folders_cache

        # Collect all folders (including subfolders) for a single mailbox.
        seen = {}
        queue = ["{}/users/{}/mailFolders".format(GRAPH_BASE, _quote_path(upn))]

        while queue:
            base_url = queue.pop(0)
            folders = self._paged_values(
                base_url,
                params={
                    "$top": "200",
                    "$select": "id,displayName,parentFolderId,childFolderCount,totalItemCount,unreadItemCount",
                    "$orderby": "displayName",
                },
            )

            for f in folders:
                fid = f.get("id")
                if not fid or fid in seen:
                    continue
                seen[fid] = f
                if int(f.get("childFolderCount") or 0) > 0:
                    queue.append(
                        "{}/users/{}/mailFolders/{}/childFolders".format(
                            GRAPH_BASE,
                            _quote_path(upn),
                            _quote_path(fid),
                        )
                    )

        folders = list(seen.values())
        self._folders_cache = folders
        self._folders_cache_expires_at = now + 30
        return folders

    def list_folder_messages(self, upn: str, folder_id: str, top=10):
        url = f"{GRAPH_BASE}/users/{_quote_path(upn)}/mailFolders/{_quote_path(folder_id)}/messages"
        params = {
            "$top": str(top),
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,from,toRecipients,receivedDateTime,bodyPreview,hasAttachments",
        }
        data = self.get_json(url, params=params)
        return data.get("value", [])

    def get_folder(self, upn: str, folder_id: str):
        url = f"{GRAPH_BASE}/users/{_quote_path(upn)}/mailFolders/{_quote_path(folder_id)}"
        params = {"$select": "id,displayName,parentFolderId"}
        return self.get_json(url, params=params)

    def get_message(self, upn: str, message_id: str):
        url = f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/{_quote_path(message_id)}"
        params = {"$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,hasAttachments"}
        headers = {"Prefer": 'outlook.body-content-type="html"'}
        return self.get_json(url, params=params, headers=headers)

    def list_attachments(self, upn: str, message_id: str):
        url = (
            f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/"
            f"{_quote_path(message_id)}/attachments"
        )
        # Note: contentId is not selectable on the base microsoft.graph.attachment type.
        # For inline-image matching we resolve contentId via per-attachment detail when needed.
        params = {"$select": "id,name,contentType,size,isInline"}
        data = self.get_json(url, params=params)
        return data.get("value", [])

    def get_attachment(self, upn: str, message_id: str, attachment_id: str):
        url = (
            f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/{_quote_path(message_id)}/attachments/{_quote_path(attachment_id)}"
        )
        # Avoid $select because attachment subtypes vary.
        return self.get_json(url)

    def download_attachment_value(self, upn: str, message_id: str, attachment_id: str) -> bytes:
        url = (
            f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/{_quote_path(message_id)}/attachments/{_quote_path(attachment_id)}/$value"
        )
        r = requests.get(url, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.content


GRAPH = GraphClient()


CHAT_PANEL_HTML = textwrap.dedent(
    """
    <div class="box chatPanel">
      <h2>Chat</h2>
      <div id="chatTranscript" class="chatTranscript" aria-label="chat transcript"></div>
      <div class="chatComposer">
        <textarea id="chatInput" class="chatInput" placeholder="Type here… (Enter to send, Shift+Enter for newline)"></textarea>
        <button id="chatSend" class="chatSend">Send</button>
      </div>
      <div id="chatStatus" class="muted">Ready</div>
    </div>
    """
).strip()

CHAT_SCRIPT_HTML = textwrap.dedent(
    r"""
    <script>
    (function(){
        const transcript = document.getElementById('chatTranscript');
        const input = document.getElementById('chatInput');
        const sendBtn = document.getElementById('chatSend');
        const status = document.getElementById('chatStatus');
        const draftKey = 'mail_client_chat_draft_v3';
        const msgsKey = 'mail_client_chat_msgs_v3';
        const respKey = 'mail_client_chat_prev_response_id_v3';

        function setStatus(t){ if(status) status.textContent = t; }

        function escapeHtml(s){
            return String(s)
              .replace(/&/g,'&amp;')
              .replace(/</g,'&lt;')
              .replace(/>/g,'&gt;')
              .replace(/"/g,'&quot;')
              .replace(/'/g,'&#39;');
        }

        function formatAssistantHtml(text){
            // Keep this intentionally simple/heuristic: highlight numeric/date-range facts
            // (commonly emitted in parentheses) with a distinct style.
            let s = escapeHtml(text || '');
            // Basic Markdown-ish bold support for **text**.
            s = s.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
            // Newlines as line breaks.
            s = s.replace(/\r\n|\r|\n/g, '<br>');
            // Highlight parenthetical facts that contain digits: (…19 emails…)
            s = s.replace(/\(([^\)]*?\d[^\)]*?)\)/g, '(<span class="chatQuant">$1</span>)');
            // Highlight common count phrases even if not parenthesized.
            s = s.replace(/\b(\d{1,3}(?:,\d{3})*)\s+(emails?|messages?|results?|attachments?)\b/gi, '<span class="chatQuant">$1 $2</span>');
            return s;
        }

        function scrollToBottom(){
            if(!transcript) return;
            transcript.scrollTop = transcript.scrollHeight;
        }

        function renderMessage(m){
            if(!transcript) return;
            const role = m && m.role ? m.role : 'assistant';
            const text = m && m.text ? m.text : '';
            const cls = role === 'user' ? 'chatMsg chatUser' : 'chatMsg chatAssistant';
            const label = role === 'user' ? 'You' : 'AI';
            const el = document.createElement('div');
            el.className = cls;
            const bodyHtml = (role === 'assistant') ? formatAssistantHtml(text) : escapeHtml(text);
            el.innerHTML = '<div class="muted" style="margin-bottom:4px">' + label + '</div>' +
                           '<div>' + bodyHtml + '</div>';

            // Click-to-resend: prior user commands can be resent without retyping.
            if(role === 'user'){
                el.title = 'Click to resend this command';
                el.style.cursor = 'pointer';
                el.addEventListener('click', function(){
                    try{
                        if(sendBtn && sendBtn.disabled) return;
                        if(!input) return;
                        input.value = String(text || '');
                        input.focus();
                        saveDraft();
                        // Let the user see the text populate before sending.
                        setTimeout(function(){ send(); }, 0);
                    }catch(e){}
                });
            }
            transcript.appendChild(el);
            scrollToBottom();
        }

        function load(){
            try{
                const d = localStorage.getItem(draftKey);
                if(d != null && input) input.value = d;
            }catch(e){}

            let msgs = [];
            try{
                const raw = localStorage.getItem(msgsKey);
                if(raw) msgs = JSON.parse(raw) || [];
            }catch(e){ msgs = []; }
            if(Array.isArray(msgs)){
                msgs.forEach(renderMessage);
            }
        }

        function saveDraft(){
            try{ localStorage.setItem(draftKey, input ? (input.value || '') : ''); }catch(e){}
        }

        function saveMsgs(msgs){
            try{ localStorage.setItem(msgsKey, JSON.stringify(msgs || [])); }catch(e){}
        }

        function getPrevResponseId(){
            try{ return localStorage.getItem(respKey) || ''; }catch(e){ return ''; }
        }

        function setPrevResponseId(v){
            try{
                if(v) localStorage.setItem(respKey, v);
                else localStorage.removeItem(respKey);
            }catch(e){}
        }

        function applySearchResults(citations, queryText){
            try{
                if(!Array.isArray(citations) || citations.length === 0) return;
                const titleEl = document.getElementById('messagesPaneTitle');
                const listEl = document.getElementById('messagesList');
                if(!listEl) return;

                if(titleEl) titleEl.textContent = 'Search results';

                // Persist the search in the URL so clicking results keeps the list.
                try{
                    const u = new URL(window.location.href);
                    u.searchParams.set('search', queryText || '');
                    // If we were previously selecting a message index, clear it.
                    u.searchParams.delete('i');
                    u.searchParams.delete('id');
                    history.replaceState({}, '', u.toString());
                }catch(e){}

                const u0 = new URL(window.location.href);
                const folder = u0.searchParams.get('folder') || '';
                const search = u0.searchParams.get('search') || (queryText || '');

                let htmlOut = '';
                for(let i=0;i<citations.length;i++){
                    const c = citations[i] || {};
                    const subj = c.subject || '(no subject)';
                    const received = c.received || '';
                    const folderPath = c.folder || '';
                    const from = c.from || '';
                    const id = c.id || '';

                    let href = '/?';
                    const parts = [];
                    if(folder) parts.push('folder=' + encodeURIComponent(folder));
                    if(search) parts.push('search=' + encodeURIComponent(search));
                    if(id) parts.push('id=' + encodeURIComponent(id));
                    href += parts.join('&');

                    const metaBits = [];
                    if(folderPath) metaBits.push(folderPath);
                    if(received) metaBits.push(received);
                    if(from) metaBits.push('From: ' + from);
                    const meta = metaBits.join(' · ');

                    htmlOut += '<li class="item">' +
                               '<a class="subject" href="' + href + '">' + escapeHtml(subj) + '</a>' +
                               (meta ? ('<div class="muted">' + escapeHtml(meta) + '</div>') : '') +
                               '</li>';
                }
                listEl.innerHTML = htmlOut || '<li class="muted">No messages returned.</li>';
            }catch(e){}
        }

        async function send(){
            if(!input) return;
            const text = (input.value || '').trim();
            if(!text){ setStatus('Type something first'); return; }

            let msgs = [];
            try{ msgs = JSON.parse(localStorage.getItem(msgsKey) || '[]') || []; }catch(e){ msgs = []; }
            msgs = Array.isArray(msgs) ? msgs : [];

            const userMsg = {role:'user', text};
            msgs.push(userMsg);
            saveMsgs(msgs);
            renderMessage(userMsg);

            setStatus('Sending…');
            if(sendBtn) sendBtn.disabled = true;
            try{
                let selectedMessageId = '';
                try{
                    const u = new URL(window.location.href);
                    selectedMessageId = u.searchParams.get('id') || '';
                }catch(e){}
                if(!selectedMessageId){
                    try{
                        const el = document.getElementById('selectedMessageId');
                        selectedMessageId = (el && el.getAttribute('data-mid')) ? (el.getAttribute('data-mid') || '') : '';
                    }catch(e){}
                }
                const payload = { message: text, previous_response_id: getPrevResponseId(), selected_message_id: selectedMessageId };
                const r = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify(payload)
                });
                const j = await r.json();
                if(!j.ok) throw new Error(j.error || 'chat failed');
                const replyText = j.reply || '';
                const assistantMsg = {role:'assistant', text: replyText};
                msgs.push(assistantMsg);
                saveMsgs(msgs);
                renderMessage(assistantMsg);
                if(j.response_id) setPrevResponseId(j.response_id);
                if(j.citations) applySearchResults(j.citations, text);
                setStatus('Ready');
                input.value = '';
                saveDraft();
            }catch(e){
                setStatus('Error');
            }finally{
                if(sendBtn) sendBtn.disabled = false;
            }
        }

        if(input){
            input.addEventListener('input', function(){ saveDraft(); setStatus('Typing…'); });
            input.addEventListener('keydown', function(ev){
                if(ev.key === 'Enter' && !ev.shiftKey){
                    ev.preventDefault();
                    send();
                }
            });
        }
        if(sendBtn){
            sendBtn.addEventListener('click', function(ev){ ev.preventDefault(); send(); });
        }

        load();
    })();
    </script>
    """
).strip()


def _cache_notify_script_html() -> str:
    enabled = "true" if _cache_notify_enabled() else "false"
    interval_s = _cache_notify_interval_seconds()
    return (
        "<script>\n"
        "(function(){\n"
        f"  const enabled = {enabled};\n"
        f"  const intervalMs = {int(interval_s) * 1000};\n"
        "  if(!enabled) return;\n"
        "  const key = 'mail_client_cache_version_v1';\n"
        "  const bar = document.getElementById('cacheNotice');\n"
        "  const reloadBtn = document.getElementById('cacheReload');\n"
        "  const dismissBtn = document.getElementById('cacheDismiss');\n"
        "  if(reloadBtn) reloadBtn.addEventListener('click', function(ev){ ev.preventDefault(); location.reload(); });\n"
        "\n"
        "  function getSeen(){ try{ return parseFloat(localStorage.getItem(key) || '0') || 0; }catch(e){ return 0; } }\n"
        "  function setSeen(v){ try{ localStorage.setItem(key, String(v || 0)); }catch(e){} }\n"
        "  function show(){ if(bar) bar.style.display = 'block'; }\n"
        "  function hide(){ if(bar) bar.style.display = 'none'; }\n"
        "\n"
        "  async function poll(){\n"
        "    try{\n"
        "      const r = await fetch('/cache/status', {cache: 'no-store'});\n"
        "      const j = await r.json();\n"
        "      const v = (j && j.version) ? parseFloat(j.version) : 0;\n"
        "      if(!v){ return; }\n"
        "      const seen = getSeen();\n"
        "      if(seen === 0){ setSeen(v); hide(); return; }\n"
        "      if(v > seen){ show(); }\n"
        "    }catch(e){}\n"
        "  }\n"
        "\n"
        "  if(dismissBtn){\n"
        "    dismissBtn.addEventListener('click', function(ev){\n"
        "      ev.preventDefault();\n"
        "      fetch('/cache/status', {cache: 'no-store'}).then(r=>r.json()).then(j=>{\n"
        "        const v = (j && j.version) ? parseFloat(j.version) : 0;\n"
        "        if(v) setSeen(v);\n"
        "        hide();\n"
        "      }).catch(()=>{ hide(); });\n"
        "    });\n"
        "  }\n"
        "\n"
        "  hide();\n"
        "  poll();\n"
        "  setInterval(poll, intervalMs);\n"
        "})();\n"
        "</script>\n"
    )


def _page(title: str, body_html: str) -> bytes:
    css = """
    html,body{height:100%;}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;margin:16px;}
    a{color:#0b57d0;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .page{display:flex;flex-direction:column;gap:12px;height:calc(100vh - 32px);}
    .headerBar{display:flex;align-items:center;justify-content:space-between;gap:12px;}
    .headerRight{margin-left:auto;display:flex;align-items:center;gap:10px;}
    .iconLink{display:inline-flex;align-items:center;gap:8px;font-size:14px;}
    .icon{font-size:18px;line-height:1;}
    .row{display:flex;gap:16px;align-items:stretch;flex:1;min-height:0;}
    .col{flex:1;min-width:0;}
    .pane{min-width:0;}
    .left{flex:0 0 320px;}
    .main{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:12px;}
    .rowInner{display:flex;gap:16px;align-items:stretch;flex:1;min-height:0;}
    .mid{flex:0 0 440px;min-height:0;}
    .right{flex:1 1 auto;}
    .chatPane{flex:0 0 360px;}
    .rightStack{display:flex;flex-direction:column;gap:12px;}
    .resizableY{resize:vertical;overflow:auto;}
    .chatBox{flex:0 0 auto;height:180px;min-height:120px;max-height:70vh;}
    .emailStack{flex:1 1 auto;min-height:0;overflow:auto;}
    .topbar{margin:0;}
    .chatPanel{display:flex;flex-direction:column;gap:8px;}
    .chatTranscript{flex:1 1 auto;min-height:120px;max-height:40vh;overflow:auto;border:1px solid #eee;border-radius:10px;padding:10px;background:#fff;}
    .chatComposer{display:flex;gap:10px;align-items:flex-end;}
    .chatInput{flex:1 1 auto;min-height:72px;}
    .chatSend{width:auto;}
    .chatMsg{border:1px solid #eee;border-radius:10px;padding:8px 10px;margin-bottom:8px;background:#fff;}
    .chatUser{background:#eef4ff;border-color:#ddd;}
    .chatAssistant{background:#f5f5f5;border-color:#ddd;}
    .chatQuant{color:#0b57d0;font-weight:700;}
    .box{border:1px solid #ddd;border-radius:10px;padding:12px;background:#fff;}
    .box h2{margin:0 0 8px 0;font-size:16px;}
    .muted{color:#666;font-size:12px;}
    .addrLine{margin-top:4px;font-size:14px;}
    .addrLabel{color:#666;}
    .addrNames{color:#0b57d0;font-weight:600;}
    .list{list-style:none;padding:0;margin:0;}
    /* Make the middle message/search results pane scroll without scrolling the whole page. */
    .mid .box{height:100%;display:flex;flex-direction:column;min-height:0;}
    #messagesList{flex:1 1 auto;min-height:0;overflow:auto;}
    .item{padding:8px;border-radius:8px;}
    .item:hover{background:#f5f5f5;}
    .selected{background:#eef4ff;}
    .subject{font-weight:600;}
    textarea{width:100%;resize:vertical;min-height:72px;padding:10px;border:1px solid #ddd;border-radius:10px;font:inherit;}
    input{width:100%;padding:10px;border:1px solid #ddd;border-radius:10px;font:inherit;}
    input[type="checkbox"]{width:auto;padding:0;border:none;border-radius:0;}
    button{padding:10px 14px;border:1px solid #ddd;border-radius:10px;background:#fff;font:inherit;cursor:pointer;}
    button:hover{background:#f5f5f5;}
    iframe{width:100%;height:70vh;border:1px solid #eee;border-radius:8px;}
    .logOutput{width:100%;min-height:160px;height:320px;margin:0;padding:10px;border:1px solid #ddd;border-radius:10px;background:#fff;overflow:auto;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;font-size:12px;}
    """

    script = """
    <script>
    (function(){
        function fallbackCopy(text){
            try{
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.setAttribute('readonly', '');
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                return true;
            }catch(e){
                return false;
            }
        }

        document.addEventListener('click', async function(ev){
            const el = (ev.target && ev.target.closest) ? ev.target.closest('.copyEmail') : null;
            if(!el) return;
            ev.preventDefault();
            const email = (el.getAttribute('data-email') || '').trim();
            if(!email) return;
            try{
                if(navigator.clipboard && navigator.clipboard.writeText){
                    await navigator.clipboard.writeText(email);
                }else{
                    fallbackCopy(email);
                }
            }catch(e){
                fallbackCopy(email);
            }
        }, true);
    })();
    </script>
        """.rstrip()

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  {body_html}
  {script}
</body>
</html>"""
    return doc.encode("utf-8")


def _folders_for_ui():
    folders = GRAPH.list_folders(MAILBOX_UPN)

    by_id = {f.get("id"): f for f in folders if f.get("id")}
    path_cache = {}

    selected_names = _cfg_get("sync", "folders", "names", default=None)
    selected_norm: set[str] | None
    if isinstance(selected_names, list) and selected_names:
        selected_norm = {str(x).strip().lower() for x in selected_names if str(x).strip()}
    else:
        selected_norm = None

    def folder_path(fid: str) -> str:
        if fid in path_cache:
            return path_cache[fid]
        f = by_id.get(fid) or {}
        name = f.get("displayName") or "(unnamed)"
        parent = f.get("parentFolderId")
        if parent and parent in by_id:
            p = folder_path(parent)
            full = f"{p} / {name}" if p else name
        else:
            full = name
        path_cache[fid] = full
        return full

    rows = []
    folder_rows = []
    for f in folders:
        fid = f.get("id")
        if not fid:
            continue
        p = folder_path(fid)
        if selected_norm is not None:
            dn = (f.get("displayName") or "").strip().lower()
            if p.lower() not in selected_norm and dn not in selected_norm:
                continue
        folder_rows.append((p, f))

    def sort_key(row):
        p, f = row
        name = (f.get("displayName") or "").strip().lower()
        parent = f.get("parentFolderId")
        # Graph sets parentFolderId to the (non-listed) MsgFolderRoot for top-level folders,
        # so "top-level" means: parent isn't another folder we know about.
        is_top_level = not parent or parent not in by_id

        # Prioritize Inbox first, then Sent, then everything else alphabetically.
        if is_top_level and name == "inbox":
            pri = 0
        elif is_top_level and name in {"sent items", "sent"}:
            pri = 1
        else:
            pri = 2
        # If both exist, show "Sent Items" before any other "Sent"-ish folder.
        sent_tie = 0 if name == "sent items" else 1
        return (pri, sent_tie, p.lower())

    for p, f in sorted(folder_rows, key=sort_key):
        fid = f["id"]
        name = (f.get("displayName") or "").strip().lower()
        parent = f.get("parentFolderId")
        is_top_level = not parent or parent not in by_id
        is_primary = is_top_level and name in {"inbox", "sent items", "sent"}

        unread = f.get("unreadItemCount")
        total = f.get("totalItemCount")
        counts = []
        if unread is not None:
            counts.append(f"{unread} unread")
        if total is not None:
            counts.append(f"{total} total")
        meta = " · ".join(counts)

        rows.append(
            {
                "id": fid,
                "path": p,
                "displayName": f.get("displayName") or p,
                "isPrimary": is_primary,
                "meta": meta,
            }
        )

    return rows


def _resolve_default_folder_id(folder_rows):
    # Prefer top-level Inbox, then top-level Sent, else first available.
    for r in folder_rows:
        if (r.get("displayName") or "").strip().lower() == "inbox":
            return r["id"]
    for r in folder_rows:
        if (r.get("displayName") or "").strip().lower() == "sent items":
            return r["id"]
    for r in folder_rows:
        if (r.get("displayName") or "").strip().lower() == "sent":
            return r["id"]
    return folder_rows[0]["id"] if folder_rows else "Inbox"


def _format_from(msg) -> str:
    try:
        if isinstance(msg, dict) and (msg.get("from_address") or msg.get("fromAddress")):
            return msg.get("from_address") or msg.get("fromAddress") or ""
        return msg.get("from", {}).get("emailAddress", {}).get("address") or ""
    except Exception:
        return ""


def _format_from_html(msg) -> str:
    try:
        name = None
        addr = None

        if isinstance(msg, dict) and (msg.get("from_name") or msg.get("from_address")):
            name = msg.get("from_name")
            addr = msg.get("from_address")
        else:
            ea = msg.get("from", {}).get("emailAddress", {})
            name = ea.get("name")
            addr = ea.get("address")

        s = _pretty_party(name, addr)
        return _copy_email_link(s, addr or "")
    except Exception:
        return ""


def _extract_cn_from_legacy_dn(value: str) -> str:
    # Legacy DN often looks like: /O=.../OU=.../CN=Recipients/CN=Some.User
    # Return the last CN segment if present.
    try:
        parts = str(value).split("/")
        cns = []
        for p in parts:
            if p.lower().startswith("cn=") and len(p) > 3:
                cns.append(p[3:])
        return (cns[-1] if cns else "").strip()
    except Exception:
        return ""


def _pretty_party(name: str | None, address: str | None) -> str:
    n = (name or "").strip()
    a = (address or "").strip()
    if n:
        return n
    if not a:
        return ""
    low = a.lower()
    if low.startswith("/o=") or low.startswith("/ou=") or low.startswith("imceaex-"):
        cn = _extract_cn_from_legacy_dn(a)
        return cn or a
    return a


def _copy_email_link(display: str, raw_email: str) -> str:
    d = (display or "").strip()
    e = (raw_email or "").strip()
    if not d and e:
        d = e
    if not d:
        return ""
    # Use <a> for native affordance; JS handler copies data-email to clipboard.
    return (
        '<a class="copyEmail addrNames" href="#" '
        + 'data-email="'
        + html.escape(e, quote=True)
        + '">'
        + html.escape(d)
        + "</a>"
    )


def _format_parties_links(parts: list[tuple[str, str]]) -> str:
    # parts: list of (display, raw_email)
    out = []
    for display, raw in parts:
        link = _copy_email_link(display, raw)
        if link:
            out.append(link)
    return ", ".join(out)


def _format_to(msg, limit: int | None = 3) -> str:
    try:
        # Cached DB shape
        if isinstance(msg, dict) and msg.get("to_json"):
            arr = json.loads(msg.get("to_json") or "[]")
            if isinstance(arr, list):
                parts = []
                take = arr if limit is None else arr[: max(0, int(limit))]
                for r in take:
                    if not isinstance(r, dict):
                        continue
                    parts.append(_pretty_party(r.get("name"), r.get("address")))
                parts = [p for p in parts if p]
                if not parts:
                    return ""
                if limit is not None and len(arr) > limit:
                    return ", ".join(parts) + f" (+{len(arr) - limit} more)"
                return ", ".join(parts)

        # Graph shape
        arr = msg.get("toRecipients") or []
        if not isinstance(arr, list):
            return ""
        parts = []
        take = arr if limit is None else arr[: max(0, int(limit))]
        for r in take:
            ea = (r or {}).get("emailAddress") or {}
            parts.append(_pretty_party(ea.get("name"), ea.get("address")))
        parts = [p for p in parts if p]
        if not parts:
            return ""
        if limit is not None and len(arr) > limit:
            return ", ".join(parts) + f" (+{len(arr) - limit} more)"
        return ", ".join(parts)
    except Exception:
        return ""


def _format_recipients_html(msg, which: str) -> str:
    # which: "to" or "cc"
    try:
        if which not in {"to", "cc"}:
            return ""

        # Cached DB
        key = "to_json" if which == "to" else "cc_json"
        if isinstance(msg, dict) and msg.get(key):
            arr = json.loads(msg.get(key) or "[]")
            if isinstance(arr, list):
                parts: list[tuple[str, str]] = []
                for r in arr:
                    if not isinstance(r, dict):
                        continue
                    display = _pretty_party(r.get("name"), r.get("address"))
                    raw = str(r.get("address") or "")
                    parts.append((display, raw))
                rendered = _format_parties_links(parts)
                if not rendered:
                    return ""
                return rendered

        # Graph
        gkey = "toRecipients" if which == "to" else "ccRecipients"
        arr = msg.get(gkey) or []
        if not isinstance(arr, list):
            return ""
        parts: list[tuple[str, str]] = []
        for r in arr:
            ea = (r or {}).get("emailAddress") or {}
            display = _pretty_party(ea.get("name"), ea.get("address"))
            raw = str(ea.get("address") or "")
            parts.append((display, raw))
        rendered = _format_parties_links(parts)
        if not rendered:
            return ""
        return rendered
    except Exception:
        return ""


def _format_dt(msg) -> str:
    raw = str(msg.get("receivedDateTime", "") or "").strip()
    if not raw:
        return ""
    try:
        v = raw
        if v.endswith("Z") and "+" not in v:
            v = v[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(v)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        d = d.astimezone(dt.timezone.utc)
        now = dt.datetime.now(dt.timezone.utc)
        days_ago = (now.date() - d.date()).days
        if days_ago < 0:
            days_ago = 0
        stamp = d.strftime("%b %d, %Y %H:%M UTC")
        return f"{stamp} ({days_ago} day{'s' if days_ago != 1 else ''} ago)"
    except Exception:
        return raw


def _db_messages_for_ui_ids(ids: list[str]) -> list[dict]:
    """Hydrate message summaries for the UI given a list of graph_id values."""
    cache_db_path = _cache_db_path()
    if not cache_db_path or not os.path.exists(cache_db_path):
        return []
    norm_ids = [str(x).strip() for x in (ids or []) if str(x).strip()]
    if not norm_ids:
        return []

    try:
        con = sqlite3.connect(cache_db_path, timeout=2)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
              m.graph_id AS id,
              m.folder_id AS folder_id,
              f.folder_path AS folder_path,
              m.subject AS subject,
              m.from_name AS from_name,
              m.from_address AS from_address,
              m.to_json AS to_json,
              m.received_dt AS receivedDateTime,
                            m.sent_dt AS sentDateTime,
              m.body_preview AS bodyPreview,
              m.has_attachments AS hasAttachments
            FROM messages m
            LEFT JOIN folders f ON f.folder_id = m.folder_id
            WHERE m.graph_id IN ("""
            + ",".join(["?"] * len(norm_ids))
            + ")",
            norm_ids,
        ).fetchall()
        con.close()
        by_id = {str(r["id"]): dict(r) for r in rows if r and str(r.get("id") or "").strip()}
        out: list[dict] = []
        for mid in norm_ids:
            d = by_id.get(mid)
            if not d:
                continue
            d["hasAttachments"] = bool(d.get("hasAttachments"))
            # If this message is in Sent Items, show sentDateTime in the UI date slot.
            try:
                fp = str(d.get("folder_path") or "").strip().lower()
                leaf = fp.split(" / ")[-1] if fp else ""
                if leaf in {"sent items", "sent"} and d.get("sentDateTime"):
                    d["receivedDateTime"] = d.get("sentDateTime")
            except Exception:
                pass
            out.append(d)
        return out
    except Exception:
        return []


def _client_view(
    folder_id: str | None,
    selected_id: str | None,
    selected_index: int | None,
    search_query: str | None = None,
) -> bytes:
    upn = MAILBOX_UPN
    folder_rows = _folders_for_ui()

    if not folder_rows:
        body = (
            "<div class=\"page\">"
            "<div class=\"box\">"
            "<div class=\"headerBar\">"
            "<div><strong>Mail</strong></div>"
            "<div class=\"headerRight\">"
            "<a class=\"iconLink\" href=\"/config\" title=\"Settings\">"
            "<span class=\"icon\">⚙</span><span>Settings</span>"
            "</a>"
            "</div></div></div>"
            "<div class=\"box\"><h2>No folders selected</h2>"
            "<div class=\"muted\">Choose one or more folders on the Settings page.</div></div>"
            "</div>"
        )
        return _page("Mail", body)

    folder_id = (folder_id or "").strip() or _resolve_default_folder_id(folder_rows)

    folders_items = []
    for r in folder_rows:
        fid = r["id"]
        href = f"/?folder={urllib.parse.quote(fid, safe='')}"
        cls = "item selected" if fid == folder_id else "item"
        label_cls = "subject" if r.get("isPrimary") else ""
        folders_items.append(
            f"<li class=\"{cls}\"><a class=\"{label_cls}\" href=\"{href}\">{html.escape(r['path'])}</a>"
            f"<div class=\"muted\">{html.escape(r.get('meta') or '')}</div></li>"
        )

    folders_pane = f"""
    <div class=\"box\">
      <h2>Mailboxes</h2>
      <ul class=\"list\">{''.join(folders_items) or '<li class="muted">No folders found.</li>'}</ul>
    </div>
    """

    search_query = (search_query or "").strip()
    if search_query:
        # Use the external search agent so search results match chat retrieval.
        agent_out = _run_search_agent_json(search_query, selected_message_id=None)
        ids: list[str] = []
        if isinstance(agent_out, dict) and agent_out.get("ok") and isinstance(agent_out.get("results"), list):
            for r in agent_out.get("results") or []:
                mid = str((r or {}).get("id") or "").strip()
                if mid:
                    ids.append(mid)
        messages = _db_messages_for_ui_ids(ids)
        folder_name = "Search results"
        is_sent_folder = False
    else:
        # Cache-only list for speed. Freshness is handled by the live poll + sync.
        messages = _db_list_messages(folder_id, top=10) or []

        try:
            folder = GRAPH.get_folder(upn, folder_id)
            folder_name = folder.get("displayName") or "Mailbox"
        except Exception:
            folder_name = "Mailbox"

        folder_name_norm = (folder_name or "").strip().lower()
        is_sent_folder = folder_name_norm in {"sent items", "sent"}

    # VS Code Simple Browser can be flaky with very long URLs.
    # Support selecting messages by index (0-9) to keep URLs short.
    # If i is provided, prefer it over any id query param (avoids stale/corrupted ids).
    if selected_index is not None:
        try:
            if 0 <= selected_index < len(messages):
                selected_id = messages[selected_index].get("id")
        except Exception:
            pass

    msg_items = []
    for idx, m in enumerate(messages):
        mid = m.get("id", "")
        subject = m.get("subject") or "(no subject)"
        dt = _format_dt(m)
        has_attachments = bool(m.get("hasAttachments"))

        # In search mode, include the search query in links so the list stays in search mode.
        if search_query:
            href = (
                f"/?folder={urllib.parse.quote(folder_id, safe='')}&search={urllib.parse.quote(search_query, safe='')}&i={idx}"
            )
        else:
            href = f"/?folder={urllib.parse.quote(folder_id, safe='')}&i={idx}"

        # If we're showing search results, decide sent-vs-inbox display per message.
        item_is_sent = is_sent_folder
        if search_query:
            fp = str(m.get("folder_path") or "").strip().lower()
            leaf = fp.split(" / ")[-1] if fp else ""
            item_is_sent = leaf in {"sent items", "sent"}

        is_selected = (selected_id and mid == selected_id) or (
            selected_index == idx and not selected_id
        )
        cls = "item selected" if is_selected else "item"
        att = " <span class=\"muted\">(attachments)</span>" if has_attachments else ""
        if item_is_sent:
            to_html = _format_recipients_html(m, "to")
            line = (
                f"<div class=\"addrLine\"><span class=\"addrLabel\">To:</span> {to_html} "
                f"<span class=\"muted\">· {html.escape(dt)}</span></div>"
            )
        else:
            from_html = _format_from_html(m)
            line = (
                f"<div class=\"addrLine\"><span class=\"addrLabel\">From:</span> {from_html} "
                f"<span class=\"muted\">· {html.escape(dt)}</span></div>"
            )

        extra = ""
        if search_query:
            fp_raw = str(m.get("folder_path") or "").strip()
            if fp_raw:
                extra = f"<div class=\"muted\">{html.escape(fp_raw)}</div>"

        msg_items.append(
            f"<li class=\"{cls}\"><a class=\"subject\" href=\"{href}\">{html.escape(subject)}</a>{att}"
            f"{line}{extra}</li>"
        )

    messages_pane = f"""
    <div class=\"box\">
            <h2 id=\"messagesPaneTitle\">{html.escape(folder_name)}</h2>
            <ul id=\"messagesList\" class=\"list\">{''.join(msg_items) or '<li class="muted">No messages returned.</li>'}</ul>
    </div>
    """

    email_parts = []

    if selected_id:
        cached_msg = _db_get_message(selected_id)
        msg = cached_msg or {}

        body_html = ""
        cached_html = (cached_msg or {}).get("bodyHtml")
        if isinstance(cached_html, str) and cached_html.strip():
            content = _rewrite_cid_images(cached_html, selected_id)
            iframe_srcdoc = html.escape(
                "<!doctype html><html><head><meta charset='utf-8'></head><body>" + content + "</body></html>",
                quote=True,
            )
            body_html = f"<iframe sandbox referrerpolicy=\"no-referrer\" srcdoc=\"{iframe_srcdoc}\"></iframe>"
        else:
            # Fall back to Graph HTML to preserve the original email formatting
            # (older DBs only have body_text).
            try:
                gmsg = GRAPH.get_message(upn, selected_id)
                msg = gmsg or msg
                body = (gmsg or {}).get("body") or {}
                content_type = (body.get("contentType") or "").lower()
                content = body.get("content") or ""
                if content_type == "html":
                    content = _rewrite_cid_images(content, selected_id)
                    iframe_srcdoc = html.escape(
                        "<!doctype html><html><head><meta charset='utf-8'></head><body>" + content + "</body></html>",
                        quote=True,
                    )
                    body_html = f"<iframe sandbox referrerpolicy=\"no-referrer\" srcdoc=\"{iframe_srcdoc}\"></iframe>"
                else:
                    body_html = f"<pre>{html.escape(content)}</pre>"
            except Exception:
                body_html = f"<pre>{html.escape((cached_msg or {}).get('bodyText') or '')}</pre>"

        to_html = _format_recipients_html(msg, "to")
        cc_html = _format_recipients_html(msg, "cc")
        to_line = (
            f"<div class=\"addrLine\"><span class=\"addrLabel\">To:</span> {to_html}</div>" if to_html else ""
        )
        cc_line = (
            f"<div class=\"addrLine\"><span class=\"addrLabel\">CC:</span> {cc_html}</div>" if cc_html else ""
        )

        from_html = _format_from_html(msg)
        from_part = (
            f"<span class=\"addrLabel\">From:</span> {from_html}"
            if from_html
            else f"From: {html.escape(_format_from(msg))}"
        )

        header = f"""
                <div class=\"box\">
                    <h2>{html.escape(msg.get('subject') or '(no subject)')}</h2>
                    <div class=\"muted\">{from_part} · Received: {html.escape(_format_dt(msg))}</div>
                    {to_line}
                    {cc_line}
                    <div style=\"margin-top:10px\">{body_html}</div>
                </div>
                """
        email_parts.append(header)

        if msg.get("hasAttachments"):
            cached = _db_list_attachments(selected_id)
            if cached:
                atts = [
                    {
                        "id": a.get("attachment_id"),
                        "name": a.get("name"),
                        "size": a.get("size"),
                        "contentType": a.get("content_type"),
                        "isInline": a.get("is_inline"),
                        "downloaded": a.get("downloaded"),
                        "hasBytes": a.get("has_bytes"),
                        "source": "local",
                    }
                    for a in cached
                ]
            else:
                atts = [dict(a, source="graph") for a in GRAPH.list_attachments(upn, selected_id)]

            att_items = []
            for a in atts:
                aid = a.get("id") or ""
                name = a.get("name") or "(unnamed)"
                size = a.get("size")
                ctype = a.get("contentType") or ""
                inline = a.get("isInline")
                source = a.get("source") or ""
                has_bytes = a.get("hasBytes")
                meta = []
                if size is not None:
                    meta.append(f"{size} bytes")
                if ctype:
                    meta.append(ctype)
                if inline:
                    meta.append("inline")
                if source == "local":
                    meta.append("local")
                    if has_bytes:
                        meta.append("cached")
                    else:
                        meta.append("no-bytes")

                href = (
                    f"/attachment?mid={urllib.parse.quote(selected_id, safe='')}&aid={urllib.parse.quote(str(aid), safe='')}"
                )

                att_items.append(
                    f"<li class=\"item\"><a class=\"subject\" href=\"{href}\">{html.escape(name)}</a>"
                    f"<div class=\"muted\">{html.escape(' · '.join(meta))}</div></li>"
                )

            att_box = f"""
            <div class=\"box\" style=\"margin-top:12px\">
              <h2>Attachments</h2>
              <ul class=\"list\">{''.join(att_items) or '<li class="muted">No attachments returned.</li>'}</ul>
            </div>
            """
            email_parts.append(att_box)
    else:
        email_parts.append(
            "<div class=\"box\"><h2>Select an email</h2><div class=\"muted\">Choose a message from the middle pane.</div></div>"
        )

    script = CHAT_SCRIPT_HTML + "\n" + _cache_notify_script_html()

    selected_marker = ""
    if selected_id:
        selected_marker = (
            '<div id="selectedMessageId" style="display:none" data-mid="'
            + html.escape(str(selected_id), quote=True)
            + '"></div>'
        )

    header = (
        "<div class=\"box\">"
        "  <div id=\"cacheNotice\" style=\"display:none;margin-bottom:10px\">"
        "    <div class=\"muted\" style=\"display:flex;align-items:center;justify-content:space-between;gap:10px\">"
        "      <div>New emails have been received. Reload the page?</div>"
        "      <div style=\"display:flex;gap:10px\">"
        "        <button id=\"cacheReload\" type=\"button\">Reload</button>"
        "        <button id=\"cacheDismiss\" type=\"button\">Dismiss</button>"
        "      </div>"
        "    </div>"
        "  </div>"
        "  <div class=\"headerBar\">"
        "    <div class=\"muted\">&nbsp;</div>"
        "    <div class=\"headerRight\">"
        "      <a class=\"iconLink\" href=\"/config\" title=\"Settings\">"
        "        <span class=\"icon\">⚙</span><span>Settings</span>"
        "      </a>"
        "    </div>"
        "  </div>"
        "</div>"
    )

    body = (
        f"<div class=\"page\">"
        f"{header}"
        f"<div class=\"row\">"
        f"<div class=\"pane left\">{folders_pane}</div>"
        f"<div class=\"main\">"
        f"<div class=\"rowInner\">"
        f"<div class=\"pane mid\">{messages_pane}</div>"
        f"<div class=\"pane right\"><div class=\"emailStack\">{''.join(email_parts)}</div></div>"
        f"<div class=\"pane chatPane\">{CHAT_PANEL_HTML}</div>"
        f"</div>"
        f"{selected_marker}"
        f"{script}"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    return _page("Mail", body)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso_z(d: dt.datetime) -> str:
    return d.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_yaml_config(path: str, cfg: dict):
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to write config. Install via APT: sudo apt install python3-yaml\n"
            f"Import error: {e}"
        )
    p = os.path.abspath(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _start_sync() -> dict:
    global SYNC_PROC, SYNC_LOG_PATH, SYNC_LOG_FH
    with SYNC_LOCK:
        if SYNC_PROC and SYNC_PROC.poll() is None:
            return {"started": False, "running": True, "logPath": SYNC_LOG_PATH}

        try:
            if SYNC_LOG_FH is not None:
                SYNC_LOG_FH.close()
        except Exception:
            pass
        SYNC_LOG_FH = None

        SYNC_LOG_PATH = f"/tmp/mailboxsync_web_{int(time.time())}.log"

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["NO_COLOR"] = "1"

        cmd = [
            "python3",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sync-app", "mailboxsync.py")),
            "--config",
            CONFIG_PATH,
            "sync",
        ]

        SYNC_LOG_FH = open(SYNC_LOG_PATH, "wb")
        SYNC_PROC = subprocess.Popen(cmd, stdout=SYNC_LOG_FH, stderr=subprocess.STDOUT, env=env)
        return {"started": True, "running": True, "logPath": SYNC_LOG_PATH}


def _auto_sync_loop():
    # Periodically run mailboxsync when enabled. This uses the existing incremental
    # coverage logic in mailboxsync, so it only fetches missing intervals.
    last_attempt = 0.0
    while not AUTO_SYNC_STOP.is_set():
        try:
            if not _auto_sync_enabled():
                AUTO_SYNC_STOP.wait(1.0)
                continue

            interval_s = float(_auto_sync_interval_seconds())
            now = time.time()
            if (now - last_attempt) < interval_s:
                AUTO_SYNC_STOP.wait(1.0)
                continue

            last_attempt = now

            with SYNC_LOCK:
                running = SYNC_PROC is not None and SYNC_PROC.poll() is None
            if running:
                AUTO_SYNC_STOP.wait(1.0)
                continue

            # Avoid unnecessary sync runs: only sync when Graph has newer mail.
            chk = _check_graph_for_new_messages()
            if chk.get("hasNew"):
                _start_sync()
        except Exception:
            # Never allow background loop to kill the server.
            AUTO_SYNC_STOP.wait(5.0)


def _read_sync_output_tail(limit_bytes: int = 64 * 1024) -> dict:
    with SYNC_LOCK:
        log_path = SYNC_LOG_PATH
        proc = SYNC_PROC
    out = ""
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                tail = min(size, int(limit_bytes))
                f.seek(size - tail)
                out = f.read().decode("utf-8", errors="replace")
        except Exception:
            out = ""
    running = False
    exit_code = None
    if proc is not None:
        rc = proc.poll()
        running = rc is None
        exit_code = rc
    return {"running": running, "exitCode": exit_code, "output": out}


def _config_view() -> bytes:
    with CONFIG_LOCK:
        cfg_path = CONFIG_PATH
    db_path = _resolve_sqlite_path() or ""

    # Build folder checkbox list from Graph.
    try:
        selected_names = _cfg_get("sync", "folders", "names", default=[]) or []
        if not isinstance(selected_names, list):
            selected_names = []
        selected_norm = {str(x).strip().lower() for x in selected_names if str(x).strip()}

        folders = GRAPH.list_folders(MAILBOX_UPN)
        by_id = {f.get("id"): f for f in folders if f.get("id")}
        path_cache: dict[str, str] = {}

        def folder_path(fid: str) -> str:
            if fid in path_cache:
                return path_cache[fid]
            f = by_id.get(fid) or {}
            name = f.get("displayName") or "(unnamed)"
            parent = f.get("parentFolderId")
            if parent and parent in by_id:
                p = folder_path(parent)
                full = f"{p} / {name}" if p else name
            else:
                full = name
            path_cache[fid] = full
            return full

        folder_rows = []
        for f in folders:
            fid = f.get("id")
            if not fid:
                continue
            p = folder_path(fid)
            folder_rows.append((p, f))

        def sort_key(row):
            p, f = row
            name = (f.get("displayName") or "").strip().lower()
            parent = f.get("parentFolderId")
            is_top_level = not parent or parent not in by_id

            # Pin top-level Inbox/Sent to the top.
            if is_top_level and name == "inbox":
                pri = 0
            elif is_top_level and name == "sent items":
                pri = 1
            elif is_top_level and name == "sent":
                pri = 2
            else:
                pri = 3
            return (pri, p.lower())

        folder_rows.sort(key=sort_key)

        folder_checks = []
        for idx, (p, f) in enumerate(folder_rows):
            name = (f.get("displayName") or "").strip()
            checked = (p.lower() in selected_norm) or (name.lower() in selected_norm)

            parent = f.get("parentFolderId")
            is_top_level = not parent or parent not in by_id
            name_norm = name.lower()
            is_primary = is_top_level and name_norm in {"inbox", "sent items", "sent"}

            cid = f"fld_{idx}"
            folder_checks.append(
                "<label style=\"display:flex;gap:10px;align-items:center;padding:6px 8px;border-radius:8px;\">"
                + f"<input type=\"checkbox\" id=\"{cid}\" name=\"syncFolder\" value=\"{html.escape(p, quote=True)}\" {'checked' if checked else ''} />"
                + f"<span class=\"{'subject' if is_primary else ''}\">{html.escape(p)}</span>"
                + ("<span class=\"muted\">Primary</span>" if is_primary else "")
                + "</label>"
            )
        folders_html = "".join(folder_checks) or '<div class="muted">No folders found.</div>'
    except Exception:
        folders_html = '<div class="muted">Unable to load folders from Graph.</div>'

    days_default = 5
    auto_enabled = bool(_cfg_get("sync", "auto_update", "enabled", default=False))
    auto_interval = int(_cfg_get("sync", "auto_update", "interval_seconds", default=60) or 60)
    auto_interval = max(15, min(3600, auto_interval))

    notify_enabled = bool(_cfg_get("web", "cache_notify", "enabled", default=True))
    notify_interval = int(_cfg_get("web", "cache_notify", "interval_seconds", default=30) or 30)
    notify_interval = max(5, min(3600, notify_interval))

    body = textwrap.dedent(
                rf"""
                <div class="page">
                    <div class="box">
                        <div class="headerBar">
                            <a class="iconLink" href="/" title="Return to email"><span class="icon">←</span><span>Return to email</span></a>
                            <div class="headerRight"></div>
                        </div>
                    </div>

                    <div class="box">
                        <h2>Settings</h2>
                        <div class="muted">Configure cache range, config.yaml path, DB path, and which folders to sync. Apply updates YAML and runs sync.</div>
                        <div style="margin-top:10px">
                            <div class="muted">Config YAML path</div>
                            <input id="cfgPath" value="{html.escape(cfg_path, quote=True)}" />
                        </div>
                        <div style="margin-top:10px">
                            <div class="muted">DB path (WSL path)</div>
                            <input id="dbPath" value="{html.escape(db_path, quote=True)}" />
                        </div>
                        <div style="margin-top:10px">
                            <div class="muted">Cache range (days)</div>
                            <input id="days" type="number" min="1" value="{days_default}" />
                        </div>
                        <div style="margin-top:10px">
                            <div class="muted">Auto-sync</div>
                            <label style="display:flex;gap:10px;align-items:center;margin-top:6px">
                                <input id="autoSync" type="checkbox" {'checked' if auto_enabled else ''} />
                                <span>Keep cache in sync automatically</span>
                            </label>
                            <div style="margin-top:6px">
                                <div class="muted">Auto-sync interval (seconds)</div>
                                <input id="autoSyncInterval" type="number" min="15" max="3600" value="{auto_interval}" />
                            </div>
                        </div>
                        <div style="margin-top:10px">
                            <div class="muted">Cache update notification</div>
                            <label style="display:flex;gap:10px;align-items:center;margin-top:6px">
                                <input id="cacheNotify" type="checkbox" {'checked' if notify_enabled else ''} />
                                <span>Notify when new emails arrive (prompt to reload)</span>
                            </label>
                            <div style="margin-top:6px">
                                <div class="muted">Notify poll interval (seconds)</div>
                                <input id="cacheNotifyInterval" type="number" min="5" max="3600" value="{notify_interval}" />
                            </div>
                        </div>
                        <div style="margin-top:10px">
                            <div class="muted">Folders to sync (and display)</div>
                            <div class="box" style="padding:8px;border-radius:10px;max-height:260px;overflow:auto">
                                {folders_html}
                            </div>
                        </div>
                        <div style="margin-top:12px;display:flex;gap:10px;align-items:center">
                            <button id="applyBtn">Apply</button>
                            <div id="applyStatus" class="muted">Ready</div>
                        </div>
                    </div>

                    <div class="box">
                        <h2>Output</h2>
                        <pre id="chatOutput" class="logOutput" aria-label="sync output">(sync output will appear here)</pre>
                    </div>

                    <script>
        (function(){{
          const btn = document.getElementById('applyBtn');
          const status = document.getElementById('applyStatus');
          const out = document.getElementById('chatOutput');
          const cfgPath = document.getElementById('cfgPath');
          const dbPath = document.getElementById('dbPath');
          const days = document.getElementById('days');
          const autoSync = document.getElementById('autoSync');
          const autoSyncInterval = document.getElementById('autoSyncInterval');
          const cacheNotify = document.getElementById('cacheNotify');
          const cacheNotifyInterval = document.getElementById('cacheNotifyInterval');
          let pollTimer = null;

          function setStatus(t){{ if(status) status.textContent = t; }}

                    function escapeHtml(s){{
                        return String(s)
                            .replace(/&/g,'&amp;')
                            .replace(/</g,'&lt;')
                            .replace(/>/g,'&gt;')
                            .replace(/"/g,'&quot;')
                            .replace(/'/g,'&#39;');
                    }}

                    function colorizeLog(s){{
                        // Keep this intentionally simple/predictable: highlight key tokens in blue.
                        let e = escapeHtml(s || '');
                        e = e.replace(/\b(Folder:)\b/g, '<span class="addrNames">$1</span>');
                        e = e.replace(/\b(sync:)\b/g, '<span class="addrNames">$1</span>');
                        e = e.replace(/\b(page=\d+)\b/g, '<span class="addrNames">$1</span>');
                        e = e.replace(/\b(time_progress=\d+(?:\.\d+)?%)\b/g, '<span class="addrNames">$1</span>');
                        e = e.replace(/\b(Traceback|ERROR|Error)\b/g, '<span class="addrNames">$1</span>');
                        return e;
                    }}

          async function poll(){{
            try{{
              const r = await fetch('/sync/status');
              const j = await r.json();
                            if(out){{
                                out.innerHTML = colorizeLog(j.output || '');
                                // Keep vscroll in sync with new log output.
                                out.scrollTop = out.scrollHeight;
                            }}
              if(j.running) setStatus('Sync running…');
              else if(j.exitCode != null) setStatus('Sync finished (exit ' + j.exitCode + ')');
              if(!j.running && pollTimer){{ clearInterval(pollTimer); pollTimer = null; }}
            }} catch(e){{}}
          }}

          async function apply(){{
            setStatus('Applying…');
            try{{
                            const folderEls = Array.from(document.querySelectorAll('input[name="syncFolder"]'));
                            const selectedFolders = folderEls.filter(x => x && x.checked).map(x => (x.value || '').trim()).filter(Boolean);
              const payload = {{
                configPath: (cfgPath && cfgPath.value) ? cfgPath.value : '',
                dbPath: (dbPath && dbPath.value) ? dbPath.value : '',
                days: (days && days.value) ? parseInt(days.value, 10) : 5,
                                folders: selectedFolders,
                                autoSyncEnabled: !!(autoSync && autoSync.checked),
                                autoSyncInterval: (autoSyncInterval && autoSyncInterval.value) ? parseInt(autoSyncInterval.value, 10) : 60,
                                                                cacheNotifyEnabled: !!(cacheNotify && cacheNotify.checked),
                                                                cacheNotifyInterval: (cacheNotifyInterval && cacheNotifyInterval.value) ? parseInt(cacheNotifyInterval.value, 10) : 30,
              }};
              const r = await fetch('/config/apply', {{
                method: 'POST',
                headers: {{'Content-Type':'application/json'}},
                body: JSON.stringify(payload)
              }});
              const j = await r.json();
              if(!j.ok) throw new Error(j.error || 'apply failed');
              setStatus(j.message || 'Applied');
              await poll();
              if(!pollTimer) pollTimer = setInterval(poll, 1000);
            }} catch(e){{ setStatus('Error'); }}
          }}

          if(btn) btn.addEventListener('click', function(ev){{ ev.preventDefault(); apply(); }});
          poll();
        }})();
        </script>
        </div>
        """
    ).strip()

    return _page("Settings", body)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # VS Code's Simple Browser is more sensitive to restrictive CSPs than Chrome.
        # Keep scripts disabled while allowing basic resources.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self' data: https: http:; "
            "script-src 'self' 'unsafe-inline'; "
            "img-src data: https: http:; "
            "style-src 'unsafe-inline' 'self' https: http:; "
            "font-src data: https: http:; "
            "frame-src 'self'; "
            "base-uri 'none';",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status=200):
        self._send(status, "text/html; charset=utf-8", body)

    def _send_text(self, text: str, status=200):
        self._send(status, "text/plain; charset=utf-8", text.encode("utf-8"))

    def _send_json(self, obj, status=200):
        self._send(status, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/chat", "/config/apply", "/agent/search"}:
                self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
                return

            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")

            if parsed.path == "/chat":
                message = str(payload.get("message") or payload.get("prompt") or "").strip()
                prev_id = str(payload.get("previous_response_id") or "").strip()
                selected_present = "selected_message_id" in payload
                selected_message_id = str(payload.get("selected_message_id") or "").strip()
                if not message:
                    self._send_json({"ok": False, "error": "empty message"}, status=HTTPStatus.BAD_REQUEST)
                    return

                model = _chat_cfg_str("chat", "model", default="gpt-5.1").strip() or "gpt-5.1"
                system_prompt = _chat_cfg_str(
                    "chat",
                    "system_prompt",
                    default=(
                        "You are a mailbox assistant. Answer using only the provided cache search results."
                    ),
                ).strip()

                # Retrieval is delegated to the external search agent CLI.
                agent_out = _run_search_agent_json(
                    message,
                    selected_message_id=(selected_message_id if selected_present else None),
                    limit=_chat_cfg_int("chat", "max_results", default=6),
                )
                if not agent_out.get("ok"):
                    self._send_json(
                        {
                            "ok": False,
                            "error": "search agent retrieval failed",
                            "detail": agent_out,
                        },
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return

                # If the agent performed a deterministic action (sync/cache status), return it directly.
                action = str(agent_out.get("action") or "").strip().lower()
                if action in {"sync_started", "sync_status", "cache_status"}:
                    direct_text = str(agent_out.get("text") or "").strip() or str(agent_out.get("title") or "").strip()
                    self._send_json({"ok": True, "reply": direct_text, "response_id": "", "citations": []})
                    return

                if not OPENAI_API_KEY:
                    self._send_json(
                        {"ok": False, "error": "OPENAI_API_KEY is not set"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                ctx_text = str(agent_out.get("context_text") or agent_out.get("text") or "").strip()
                selected_ctx = str(agent_out.get("selected_context_text") or "").strip()
                citations = agent_out.get("citations") if isinstance(agent_out.get("citations"), list) else []
                notices_text = str(agent_out.get("notices_text") or "").strip()

                composed_parts = [system_prompt]
                composed_parts.append(
                    "\n\nRESPONSE STYLE (important):\n"
                    "- Answer the user's question qualitatively and concisely.\n"
                    "- Do NOT paste or restate the raw cache search results list.\n"
                    "- If the user asks to *list* emails (e.g. 'show me emails I got today'), do NOT enumerate them in chat; instead say how many you found and tell the user to use the Search results pane.\n"
                    "- If you need to reference a specific email, cite it by subject (and optionally #N), not by dumping headers/ids.\n"
                )
                composed_parts.append("\n\nCACHE SEARCH RESULTS (search agent / local SQLite):\n" + (ctx_text or "(No results.)"))
                if notices_text:
                    composed_parts.append("\n\nCACHE NOTICES (search agent):\n" + notices_text)
                if selected_ctx:
                    composed_parts.append("\n\nSELECTED EMAIL ATTACHMENTS (local SQLite):\n" + selected_ctx)
                composed_parts.append("\n\nUSER QUESTION:\n" + message)
                composed = "".join(composed_parts)

                req = {
                    "model": model,
                    "input": composed,
                }
                if prev_id:
                    req["previous_response_id"] = prev_id

                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                r = requests.post(
                    "https://api.openai.com/v1/responses",
                    headers=headers,
                    json=req,
                    timeout=90,
                )
                if r.status_code >= 400:
                    self._send_json(
                        {"ok": False, "error": f"OpenAI error {r.status_code}", "detail": r.text},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return
                data = r.json() if r.content else {}

                out = []
                for item in data.get("output", []) or []:
                    for c in item.get("content", []) or []:
                        if c.get("type") == "output_text":
                            out.append(c.get("text", ""))
                reply = "\n".join(out).strip()

                self._send_json(
                    {"ok": True, "reply": reply, "response_id": data.get("id"), "citations": citations}
                )
                return

            if parsed.path == "/agent/search":
                q = str(payload.get("query") or payload.get("message") or payload.get("q") or "").strip()
                selected_message_id = str(payload.get("selected_message_id") or payload.get("selected") or "").strip() or None
                try:
                    lim = int(payload.get("limit")) if payload.get("limit") is not None else None
                except Exception:
                    lim = None

                out = _run_search_agent_json(q, selected_message_id=selected_message_id, limit=lim)
                status = HTTPStatus.OK if out.get("ok") else HTTPStatus.BAD_REQUEST
                self._send_json(out, status=int(status))
                return

            if parsed.path == "/config/apply":
                new_cfg_path = str(payload.get("configPath") or "").strip() or CONFIG_PATH
                db_path_raw = str(payload.get("dbPath") or "").strip()
                folders_raw = payload.get("folders")
                auto_enabled = bool(payload.get("autoSyncEnabled"))
                notify_enabled = bool(payload.get("cacheNotifyEnabled"))
                try:
                    auto_interval = int(payload.get("autoSyncInterval") or 60)
                except Exception:
                    auto_interval = 60
                auto_interval = max(15, min(3600, auto_interval))

                try:
                    notify_interval = int(payload.get("cacheNotifyInterval") or 30)
                except Exception:
                    notify_interval = 30
                notify_interval = max(5, min(3600, notify_interval))

                try:
                    days = int(payload.get("days") or 5)
                except Exception:
                    days = 5
                days = max(1, days)

                # Base config: load from the new path if it exists, else start from current CFG.
                base_cfg = _load_yaml_config(new_cfg_path) if os.path.exists(new_cfg_path) else CFG
                if not isinstance(base_cfg, dict):
                    base_cfg = {}

                base_cfg.setdefault("sync", {})
                base_cfg["sync"].setdefault("cache_range", {})
                base_cfg.setdefault("storage", {})

                start_dt = _utc_now() - dt.timedelta(days=days)
                base_cfg["sync"]["cache_range"]["start_date"] = _iso_z(start_dt)
                base_cfg["sync"]["cache_range"]["end_date"] = "now"

                # Avoid "test" settings that make the cache appear empty.
                try:
                    cur_max = int(base_cfg["sync"].get("max_messages_per_run") or 0)
                except Exception:
                    cur_max = 0
                if cur_max < 1000:
                    base_cfg["sync"]["max_messages_per_run"] = 500000

                if db_path_raw:
                    base_cfg["storage"]["sqlite_path"] = db_path_raw

                # Selected folders (paths/names) to sync and display.
                if folders_raw is not None:
                    if not isinstance(folders_raw, list):
                        self._send_json({"ok": False, "error": "folders must be a list"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    cleaned = [str(x).strip() for x in folders_raw if str(x).strip()]
                    if not cleaned:
                        self._send_json(
                            {"ok": False, "error": "Select at least one folder to sync"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    base_cfg.setdefault("sync", {})
                    base_cfg["sync"].setdefault("folders", {})
                    base_cfg["sync"]["folders"]["names"] = cleaned

                # Auto-sync settings
                base_cfg.setdefault("sync", {})
                base_cfg["sync"].setdefault("auto_update", {})
                base_cfg["sync"]["auto_update"]["enabled"] = bool(auto_enabled)
                base_cfg["sync"]["auto_update"]["interval_seconds"] = int(auto_interval)

                # Cache update notification settings
                base_cfg.setdefault("web", {})
                base_cfg["web"].setdefault("cache_notify", {})
                base_cfg["web"]["cache_notify"]["enabled"] = bool(notify_enabled)
                base_cfg["web"]["cache_notify"]["interval_seconds"] = int(notify_interval)

                # Write + reload.
                _write_yaml_config(new_cfg_path, base_cfg)
                _reload_config(new_cfg_path)

                sync_info = _start_sync()
                msg = "Applied"
                if sync_info.get("started"):
                    msg += "; sync started"
                elif sync_info.get("running"):
                    msg += "; sync already running"

                self._send_json(
                    {
                        "ok": True,
                        "message": msg,
                        "configPath": CONFIG_PATH,
                        "dbPath": _resolve_sqlite_path(),
                        "sync": sync_info,
                    }
                )
                return
        except Exception:
            self._send_text(traceback.format_exc(), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path == "/healthz":
                self._send_text("ok")
                return

            if path == "/config":
                self._send_html(_config_view())
                return

            if path == "/sync/status":
                self._send_json(_read_sync_output_tail())
                return

            if path == "/cache/status":
                with SYNC_LOCK:
                    running = SYNC_PROC is not None and SYNC_PROC.poll() is None
                self._send_json({"ok": True, "version": _cache_version(), "syncRunning": bool(running)})
                return

            if path == "/inline":
                mid = (qs.get("mid") or [""])[0]
                cid_raw = (qs.get("cid") or [""])[0]
                if not mid or not cid_raw:
                    self._send_text("Bad Request", status=HTTPStatus.BAD_REQUEST)
                    return

                cid = _normalize_content_id(cid_raw)
                if not cid:
                    self._send_text("Bad Request", status=HTTPStatus.BAD_REQUEST)
                    return

                # 1) Try DB: find an inline attachment with matching content_id
                cache_db_path = _cache_db_path()
                if cache_db_path and os.path.exists(cache_db_path) and _db_has_column(cache_db_path, "attachments", "content_id"):
                    try:
                        con = sqlite3.connect(cache_db_path, timeout=1)
                        con.row_factory = sqlite3.Row
                        rows = con.execute(
                            """
                            SELECT attachment_id, name, content_type, size, content_bytes
                            FROM attachments
                            WHERE message_graph_id = ? AND is_inline = 1 AND content_id IS NOT NULL
                            """,
                            (mid,),
                        ).fetchall()
                        con.close()
                        for r in rows:
                            if _normalize_content_id(r["content_id"]) == cid:
                                b = r["content_bytes"]
                                ctype = r["content_type"] or "application/octet-stream"
                                if b is None:
                                    # bytes missing; let Graph fallback handle download
                                    break
                                self.send_response(HTTPStatus.OK)
                                self.send_header("Content-Type", str(ctype) or "application/octet-stream")
                                self.send_header("Cache-Control", "no-store")
                                self.send_header("X-Content-Type-Options", "nosniff")
                                self.end_headers()
                                self.wfile.write(b)
                                return
                    except Exception:
                        pass

                # 2) Graph fallback: locate by contentId then download
                try:
                    atts = GRAPH.list_attachments(MAILBOX_UPN, mid) or []
                    match = None
                    for a in atts:
                        if not a.get("isInline"):
                            continue
                        aid = a.get("id")
                        if not aid:
                            continue
                        # contentId is not present on the list response; fetch detail.
                        try:
                            detail = GRAPH.get_attachment(MAILBOX_UPN, mid, aid) or {}
                        except Exception:
                            detail = {}
                        if _normalize_content_id(detail.get("contentId")) == cid:
                            match = dict(a)
                            match["contentId"] = detail.get("contentId")
                            break
                    if not match:
                        self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
                        return
                    aid = match.get("id")
                    if not aid:
                        self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
                        return
                    ctype = match.get("contentType") or "application/octet-stream"
                    b = GRAPH.download_attachment_value(MAILBOX_UPN, mid, aid)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", str(ctype) or "application/octet-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(b)
                    return
                except Exception:
                    self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
                    return

            if path == "/attachment":
                mid = (qs.get("mid") or [""])[0]
                aid = (qs.get("aid") or [""])[0]
                if not mid or not aid:
                    self._send_text("Bad Request", status=HTTPStatus.BAD_REQUEST)
                    return

                # 1) Prefer local DB bytes
                b, meta = _db_get_attachment_bytes(mid, aid)
                name = (meta or {}).get("name") or "attachment"
                ctype = (meta or {}).get("content_type") or "application/octet-stream"
                size = (meta or {}).get("size")

                # 2) Fall back to Graph if bytes missing
                if b is None:
                    try:
                        detail = GRAPH.get_attachment(MAILBOX_UPN, mid, aid)
                        name = detail.get("name") or name
                        ctype = detail.get("contentType") or ctype
                        size = detail.get("size") or size
                        cb = detail.get("contentBytes")
                        if cb:
                            b = base64.b64decode(cb)
                        else:
                            b = GRAPH.download_attachment_value(MAILBOX_UPN, mid, aid)

                        if _sync_attachments_mode() == "full":
                            _db_upsert_attachment_bytes(
                                mid,
                                aid,
                                name=name,
                                content_type=ctype,
                                size=int(size) if size is not None else None,
                                content_bytes=b,
                                downloaded=1,
                                error=None,
                            )
                    except Exception as e:
                        if _sync_attachments_mode() == "full":
                            _db_upsert_attachment_bytes(
                                mid,
                                aid,
                                name=name,
                                content_type=ctype,
                                size=int(size) if size is not None else None,
                                content_bytes=None,
                                downloaded=0,
                                error=f"download failed: {e}",
                            )
                        raise

                if b is None:
                    self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
                    return

                # Stream bytes
                safe_name = "".join(ch for ch in str(name) if ch not in "\r\n\x00").strip() or "attachment"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", str(ctype) or "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(b)
                return

            if path == "/":
                folder_id = (qs.get("folder") or [None])[0]
                selected_id = (qs.get("id") or [None])[0]
                search_query = (qs.get("search") or [None])[0]
                i_raw = (qs.get("i") or [None])[0]
                selected_index = None
                if i_raw is not None:
                    try:
                        selected_index = int(i_raw)
                    except Exception:
                        selected_index = None

                self._send_html(_client_view(folder_id, selected_id, selected_index, search_query=search_query))
                return

            if path == "/inbox":
                # Backwards-compatible alias.
                selected_id = (qs.get("id") or [None])[0]
                search_query = (qs.get("search") or [None])[0]
                i_raw = (qs.get("i") or [None])[0]
                selected_index = None
                if i_raw is not None:
                    try:
                        selected_index = int(i_raw)
                    except Exception:
                        selected_index = None

                self._send_html(_client_view("Inbox", selected_id, selected_index, search_query=search_query))
                return

            if path == "/folder":
                folder_id = (qs.get("folder") or [""])[0].strip() or "Inbox"
                selected_id = (qs.get("id") or [None])[0]
                search_query = (qs.get("search") or [None])[0]
                i_raw = (qs.get("i") or [None])[0]
                selected_index = None
                if i_raw is not None:
                    try:
                        selected_index = int(i_raw)
                    except Exception:
                        selected_index = None

                self._send_html(_client_view(folder_id, selected_id, selected_index, search_query=search_query))
                return

            self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.text
            except Exception:
                pass
            self._send_text(f"Graph error: {e}\n\n{detail}", status=HTTPStatus.BAD_GATEWAY)
        except Exception:
            self._send_text(traceback.format_exc(), status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    host = os.environ.get("HOST") or str(
        _cfg_get("web", "host", default=_cfg_get("mailweb", "host", default="127.0.0.1"))
    )
    # Some WSL/Windows setups block binding to certain well-known dev ports (e.g. 8000).
    # Default to 8001 which tends to be available.
    port = int(
        os.environ.get("PORT")
        or str(_cfg_get("web", "port", default=_cfg_get("mailweb", "port", default="8001")))
    )

    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        # Common dev-loop error: server already running.
        if getattr(e, "errno", None) in {98, 48}:
            raise SystemExit(
                f"Port {port} is already in use. "
                f"If the server is already running, open http://{host}:{port} in your browser. "
                f"Otherwise stop the old process and retry."
            )
        raise

    print(f"Serving on http://{host}:{port}")
    t = threading.Thread(target=_auto_sync_loop, name="auto-sync", daemon=True)
    t.start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
