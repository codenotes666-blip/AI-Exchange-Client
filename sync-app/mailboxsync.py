#!/usr/bin/env python3
import argparse
import datetime as dt
import html
import json
import os
import re
import sqlite3
import base64
import subprocess
import sys
import tempfile
import time
import urllib.parse

import msal
import requests


GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SCOPE = ["https://graph.microsoft.com/.default"]


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    force = (os.environ.get("FORCE_COLOR") or "").strip().lower()
    if force in {"1", "true", "yes", "y", "on"}:
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _c(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _green(text: str) -> str:
    return _c(text, "32")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _blue(text: str) -> str:
    return _c(text, "34")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str) -> dt.datetime:
    # Accept Graph-style timestamps: 2025-11-01T00:00:00Z
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)


def _quote_path(segment: str) -> str:
    return urllib.parse.quote(segment, safe="")


def _strip_html_to_text(content: str) -> str:
    # Minimal HTML-to-text; good enough for initial indexing without extra deps.
    # - remove scripts/styles
    # - convert <br>/<p>/<div>/<li> to newlines
    # - strip tags
    # - unescape entities
    if not content:
        return ""
    s = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", content)
    s = re.sub(r"(?i)<br\\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|li|tr|h[1-6])>", "\n", s)
    s = re.sub(r"(?is)<.*?>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _is_text_like_attachment(content_type: str) -> bool:
    ct = (content_type or "").lower().strip()
    return ct.startswith("text/") or ct in {
        "application/json",
        "application/xml",
        "application/x-xml",
        "application/x-www-form-urlencoded",
        "application/yaml",
        "application/x-yaml",
    }


def _bytes_to_text_best_effort(b: bytes, limit: int = 1024 * 1024) -> str:
    if not b or len(b) > limit:
        return ""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc, errors="replace")
        except Exception:
            continue
    return ""


def _is_pdf_attachment(name: str | None, content_type: str | None) -> bool:
    ct = (content_type or "").lower().strip()
    if ct == "application/pdf" or ct.endswith("+pdf"):
        return True
    n = (name or "").lower().strip()
    return n.endswith(".pdf")


def _pdf_bytes_to_text_best_effort(b: bytes, *, limit: int = 12 * 1024 * 1024) -> str:
    """Best-effort PDF text extraction.

    Tries, in order:
    - system `pdftotext` (poppler)
    - pure-Python `pypdf` (if installed)

    Returns '' on failure.
    """
    if not b or len(b) > limit:
        return ""

    # 1) Prefer the system binary when available (fast, good quality).
    try:
        cmd = ["pdftotext", "-nopgbrk", "-layout"]
        # Verify availability.
        try:
            subprocess.run(["pdftotext", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise FileNotFoundError()
        with tempfile.NamedTemporaryFile(prefix="mailboxsync_", suffix=".pdf", delete=True) as f_in:
            f_in.write(b)
            f_in.flush()
            with tempfile.NamedTemporaryFile(prefix="mailboxsync_", suffix=".txt", delete=True) as f_out:
                r = subprocess.run(
                    cmd + [f_in.name, f_out.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                if r.returncode != 0:
                    return ""
                try:
                    out_b = f_out.read()
                except Exception:
                    out_b = b""
        return _bytes_to_text_best_effort(out_b, limit=2 * 1024 * 1024)
    except FileNotFoundError:
        pass
    except Exception:
        # Fall through to pypdf.
        pass

    # 2) Fallback: pypdf (pure Python, no external deps).
    try:
        from io import BytesIO

        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return ""

        reader = PdfReader(BytesIO(b))
        parts: list[str] = []
        # Extract text from a bounded number of pages to keep runtime sane.
        max_pages = 40
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
        text = "\n\n".join(parts).strip()
        # Normalize a bit.
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Limit stored text size.
        if len(text) > 2 * 1024 * 1024:
            text = text[: 2 * 1024 * 1024]
        return text
    except Exception:
        return ""


class GraphClient:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        self._app = msal.ConfidentialClientApplication(
            client_id,
            authority=authority,
            client_credential=client_secret,
        )
        self._access_token = None
        self._access_token_expires_at = 0

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

    def _paged_values_iter(self, url: str, *, params=None, headers=None, timeout=30, on_page=None):
        """Yield values across pages; optionally call on_page with progress metadata.

        Graph list endpoints usually do not provide a total count by default.
        This iterator provides user-visible progress based on page fetches.
        """

        next_url = url
        next_params = params
        page = 0
        total = 0
        t0 = time.time()

        while next_url:
            page += 1
            data = self.get_json(next_url, params=next_params, headers=headers, timeout=timeout)
            vals = data.get("value", []) or []
            total += len(vals)
            next_link = data.get("@odata.nextLink")
            if on_page:
                try:
                    on_page(
                        {
                            "page": page,
                            "total": total,
                            "page_count": len(vals),
                            "elapsed_s": time.time() - t0,
                            "last": vals[-1] if vals else None,
                            "has_next": bool(next_link),
                        }
                    )
                except Exception:
                    # Progress callback must never break the sync.
                    pass
            for v in vals:
                yield v
            next_url = next_link
            next_params = None

    def list_folders(self, upn: str):
        seen = {}
        queue = [f"{GRAPH_BASE}/users/{_quote_path(upn)}/mailFolders"]

        while queue:
            base_url = queue.pop(0)
            folders = self._paged_values(
                base_url,
                params={
                    "$top": "200",
                    "$select": "id,displayName,parentFolderId,childFolderCount",
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
                        f"{GRAPH_BASE}/users/{_quote_path(upn)}/mailFolders/{_quote_path(fid)}/childFolders"
                    )

        return list(seen.values())

    def list_messages_in_range(self, upn: str, folder_id: str, *, start_iso: str, end_iso: str, page_size: int):
        url = (
            f"{GRAPH_BASE}/users/{_quote_path(upn)}/mailFolders/{_quote_path(folder_id)}/messages"
        )
        # Use receivedDateTime range. Use 'lt' for end to avoid overlap issues.
        filt = f"receivedDateTime ge {start_iso} and receivedDateTime lt {end_iso}"
        params = {
            "$top": str(page_size),
            "$orderby": "receivedDateTime asc",
            "$filter": filt,
            "$select": ",".join(
                [
                    "id",
                    "subject",
                    "from",
                    "toRecipients",
                    "ccRecipients",
                    "bccRecipients",
                    "receivedDateTime",
                    "sentDateTime",
                    "bodyPreview",
                    "body",
                    "conversationId",
                    "internetMessageId",
                    "hasAttachments",
                    "lastModifiedDateTime",
                ]
            ),
        }
        headers = {"Prefer": 'outlook.body-content-type="html"'}
        return self._paged_values(url, params=params, headers=headers)

    def iter_messages_in_range(
        self,
        upn: str,
        folder_id: str,
        *,
        start_iso: str,
        end_iso: str,
        page_size: int,
        on_page=None,
    ):
        url = f"{GRAPH_BASE}/users/{_quote_path(upn)}/mailFolders/{_quote_path(folder_id)}/messages"
        filt = f"receivedDateTime ge {start_iso} and receivedDateTime lt {end_iso}"
        params = {
            "$top": str(page_size),
            "$orderby": "receivedDateTime asc",
            "$filter": filt,
            "$select": ",".join(
                [
                    "id",
                    "subject",
                    "from",
                    "toRecipients",
                    "ccRecipients",
                    "bccRecipients",
                    "receivedDateTime",
                    "sentDateTime",
                    "bodyPreview",
                    "body",
                    "conversationId",
                    "internetMessageId",
                    "hasAttachments",
                    "lastModifiedDateTime",
                ]
            ),
        }
        headers = {"Prefer": 'outlook.body-content-type="html"'}
        return self._paged_values_iter(url, params=params, headers=headers, on_page=on_page)

    def list_attachments(self, upn: str, message_id: str):
        url = f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/{_quote_path(message_id)}/attachments"
        return self._paged_values(
            url,
            params={
                "$top": "200",
                "$select": ",".join(
                    [
                        "id",
                        "name",
                        "contentType",
                        "size",
                        "isInline",
                    ]
                ),
            },
        )

    def get_attachment(self, upn: str, message_id: str, attachment_id: str):
        url = (
            f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/{_quote_path(message_id)}/attachments/{_quote_path(attachment_id)}"
        )
        # Do not use $select here: attachment subtypes (fileAttachment, itemAttachment, referenceAttachment)
        # have different properties and Graph can error if selecting properties not on the concrete type.
        return self.get_json(url)

    def download_attachment_value(self, upn: str, message_id: str, attachment_id: str) -> bytes:
        url = (
            f"{GRAPH_BASE}/users/{_quote_path(upn)}/messages/{_quote_path(message_id)}/attachments/{_quote_path(attachment_id)}/$value"
        )
        r = requests.get(url, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.content


class Db:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def close(self):
        self.conn.close()

    def init_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY,
                folder_id TEXT NOT NULL,
                folder_path TEXT NOT NULL,
                UNIQUE(folder_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                graph_id TEXT NOT NULL,
                folder_id TEXT NOT NULL,
                received_dt TEXT,
                sent_dt TEXT,
                last_modified_dt TEXT,
                subject TEXT,
                from_name TEXT,
                from_address TEXT,
                to_json TEXT,
                cc_json TEXT,
                bcc_json TEXT,
                conversation_id TEXT,
                internet_message_id TEXT,
                body_preview TEXT,
                body_text TEXT,
                body_html TEXT,
                has_attachments INTEGER,
                UNIQUE(graph_id)
            );

            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY,
                message_graph_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                attachment_type TEXT,
                name TEXT,
                content_type TEXT,
                size INTEGER,
                is_inline INTEGER,
                content_id TEXT,
                last_modified_dt TEXT,
                content_bytes BLOB,
                content_text TEXT,
                downloaded INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                UNIQUE(message_graph_id, attachment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_attachments_message_graph_id ON attachments(message_graph_id);

            CREATE TABLE IF NOT EXISTS coverage (
                id INTEGER PRIMARY KEY,
                folder_id TEXT NOT NULL,
                start_dt TEXT NOT NULL,
                end_dt TEXT NOT NULL,
                UNIQUE(folder_id, start_dt, end_dt)
            );

            -- FTS5 full-text index over subject + body_text
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_messages USING fts5(
                subject,
                body_text,
                content='messages',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO fts_messages(rowid, subject, body_text)
                VALUES (new.id, coalesce(new.subject,''), coalesce(new.body_text,''));
            END;

            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO fts_messages(fts_messages, rowid, subject, body_text)
                VALUES('delete', old.id, old.subject, old.body_text);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO fts_messages(fts_messages, rowid, subject, body_text)
                VALUES('delete', old.id, old.subject, old.body_text);
                INSERT INTO fts_messages(rowid, subject, body_text)
                VALUES (new.id, coalesce(new.subject,''), coalesce(new.body_text,''));
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_attachments USING fts5(
                name,
                content_text,
                content='attachments',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS attachments_ai AFTER INSERT ON attachments BEGIN
                INSERT INTO fts_attachments(rowid, name, content_text)
                VALUES (new.id, coalesce(new.name,''), coalesce(new.content_text,''));
            END;

            CREATE TRIGGER IF NOT EXISTS attachments_ad AFTER DELETE ON attachments BEGIN
                INSERT INTO fts_attachments(fts_attachments, rowid, name, content_text)
                VALUES('delete', old.id, old.name, old.content_text);
            END;

            CREATE TRIGGER IF NOT EXISTS attachments_au AFTER UPDATE ON attachments BEGIN
                INSERT INTO fts_attachments(fts_attachments, rowid, name, content_text)
                VALUES('delete', old.id, old.name, old.content_text);
                INSERT INTO fts_attachments(rowid, name, content_text)
                VALUES (new.id, coalesce(new.name,''), coalesce(new.content_text,''));
            END;
            """
        )
        self.conn.commit()

        # Lightweight migration for older DBs.
        try:
            cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(messages)").fetchall()}
            if "body_html" not in cols:
                self.conn.execute("ALTER TABLE messages ADD COLUMN body_html TEXT")
                self.conn.commit()
        except Exception:
            # If migration fails, keep going; text-only cache will still work.
            pass

    def upsert_message(self, row: dict) -> int:
        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join(cols)
        update_list = ",".join([f"{c}=excluded.{c}" for c in cols if c != "graph_id"])

        sql = (
            f"INSERT INTO messages ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(graph_id) DO UPDATE SET {update_list}"
        )
        self.conn.execute(sql, [row[c] for c in cols])
        found = self.conn.execute("SELECT id FROM messages WHERE graph_id=?", (row["graph_id"],)).fetchone()
        return int(found["id"]) if found else 0

    def upsert_attachment(self, row: dict):
        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join(cols)
        update_list = ",".join(
            [f"{c}=excluded.{c}" for c in cols if c not in {"message_graph_id", "attachment_id"}]
        )
        sql = (
            f"INSERT INTO attachments ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(message_graph_id, attachment_id) DO UPDATE SET {update_list}"
        )
        self.conn.execute(sql, [row[c] for c in cols])

    def get_attachments_for_message(self, message_graph_id: str):
        rows = self.conn.execute(
            "SELECT * FROM attachments WHERE message_graph_id=? ORDER BY name ASC", (message_graph_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_coverage(self, folder_id: str):
        rows = self.conn.execute(
            "SELECT start_dt, end_dt FROM coverage WHERE folder_id=? ORDER BY start_dt ASC",
            (folder_id,),
        ).fetchall()
        return [(r["start_dt"], r["end_dt"]) for r in rows]

    def upsert_folder(self, folder_id: str, folder_path: str):
        self.conn.execute(
            """
            INSERT INTO folders(folder_id, folder_path) VALUES (?, ?)
            ON CONFLICT(folder_id) DO UPDATE SET folder_path=excluded.folder_path
            """,
            (folder_id, folder_path),
        )

    def set_coverage(self, folder_id: str, intervals: list[tuple[str, str]]):
        self.conn.execute("DELETE FROM coverage WHERE folder_id=?", (folder_id,))
        self.conn.executemany(
            "INSERT INTO coverage(folder_id, start_dt, end_dt) VALUES (?,?,?)",
            [(folder_id, s, e) for (s, e) in intervals],
        )


def _merge_intervals(intervals: list[tuple[dt.datetime, dt.datetime]]) -> list[tuple[dt.datetime, dt.datetime]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        # Merge if overlapping or adjacent (<=1s gap)
        if s <= pe + dt.timedelta(seconds=1):
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _missing_intervals(
    want: tuple[dt.datetime, dt.datetime], have: list[tuple[dt.datetime, dt.datetime]]
) -> list[tuple[dt.datetime, dt.datetime]]:
    ws, we = want
    if ws >= we:
        return []
    have = _merge_intervals([(max(ws, s), min(we, e)) for s, e in have if e > ws and s < we])
    missing = []
    cur = ws
    for s, e in have:
        if cur < s:
            missing.append((cur, s))
        cur = max(cur, e)
    if cur < we:
        missing.append((cur, we))
    return missing


def _load_config(path: str) -> dict:
    # YAML requested. Prefer PyYAML if present (APT: python3-yaml).
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read the YAML config. "
            "Install via APT: sudo apt install python3-yaml\n"
            f"Import error: {e}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def _env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _mailboxsync(config_path: str, *, sqlite_override: str | None = None):
    cfg = _load_config(config_path)

    tenant_env = cfg.get("graph", {}).get("tenant_id_env")
    client_env = cfg.get("graph", {}).get("client_id_env")
    secret_env = cfg.get("graph", {}).get("client_secret_env")
    upn_env = cfg.get("mailbox", {}).get("mailbox_upn_env")

    if not all([tenant_env, client_env, secret_env, upn_env]):
        raise RuntimeError("Config missing required graph/mailbox env var names")

    tenant_id = _env_required(str(tenant_env))
    client_id = _env_required(str(client_env))
    client_secret = _env_required(str(secret_env))
    upn = _env_required(str(upn_env))

    cache_range = cfg.get("sync", {}).get("cache_range", {})
    start_raw = str(cache_range.get("start_date"))
    end_raw = str(cache_range.get("end_date"))

    if not start_raw or start_raw == "None":
        raise RuntimeError("sync.cache_range.start_date is required")
    if not end_raw or end_raw == "None":
        raise RuntimeError("sync.cache_range.end_date is required")

    start_dt = _parse_iso_utc(start_raw)
    end_dt = _parse_iso_utc(_utc_now_iso() if end_raw.lower() == "now" else end_raw)

    if start_dt >= end_dt:
        raise RuntimeError("cache_range start_date must be < end_date")

    folders_cfg = cfg.get("sync", {}).get("folders", {})
    folder_names = folders_cfg.get("names")
    if not isinstance(folder_names, list) or not folder_names:
        raise RuntimeError("sync.folders.names must be a non-empty list")

    # Only Inbox for now (per current config), but code supports multiple.
    folder_names_norm = [str(x).strip() for x in folder_names if str(x).strip()]

    if sqlite_override:
        sqlite_path = os.path.abspath(sqlite_override)
    else:
        storage = cfg.get("storage", {})
        sqlite_path = str(storage.get("sqlite_path") or "./mail_cache.sqlite")
        # Resolve relative to config directory.
        sqlite_path = os.path.abspath(os.path.join(os.path.dirname(config_path), sqlite_path))

    max_messages = int(cfg.get("sync", {}).get("max_messages_per_run") or 5000)
    page_size = int(cfg.get("sync", {}).get("page_size") or 50)

    attachments_mode = (
        (cfg.get("sync", {}).get("attachments", {}) or {}).get("mode")
        or "full"
    )
    attachments_mode = str(attachments_mode).strip().lower()
    if attachments_mode not in {"full", "ids"}:
        raise RuntimeError("sync.attachments.mode must be one of: full, ids")

    print(f"sync: mailbox={upn}")
    print(f"sync: cache_range={start_dt.isoformat()}Z..{end_dt.isoformat()}Z")
    print(f"sync: db={sqlite_path}")
    print(f"sync: attachments_mode={attachments_mode}")

    graph = GraphClient(tenant_id, client_id, client_secret)
    db = Db(sqlite_path)
    try:
        db.init_schema()

        folders = graph.list_folders(upn)
        by_id = {f.get("id"): f for f in folders if f.get("id")}

        def folder_path(fid: str) -> str:
            f = by_id.get(fid) or {}
            name = f.get("displayName") or "(unnamed)"
            parent = f.get("parentFolderId")
            if parent and parent in by_id:
                return folder_path(parent) + " / " + name
            return name

        # Resolve folder_ids for requested names/paths.
        wanted = {n.lower(): n for n in folder_names_norm}
        resolved = []
        for f in folders:
            fid = f.get("id")
            if not fid:
                continue
            p = folder_path(fid)
            dn = (f.get("displayName") or "").strip()
            if dn.lower() in wanted or p.lower() in wanted:
                resolved.append((fid, p))

        if not resolved:
            raise RuntimeError(f"Could not resolve folders from config: {folder_names_norm}")

        for folder_id, folder_p in resolved:
            db.upsert_folder(folder_id, folder_p)
        db.conn.commit()

        total_inserted = 0
        for folder_id, folder_p in resolved:
            have = [( _parse_iso_utc(s), _parse_iso_utc(e)) for s,e in db.get_coverage(folder_id)]
            want = (start_dt, end_dt)
            missing = _missing_intervals(want, have)
            print(f"{_yellow('Folder:')} {_yellow(folder_p)} ({folder_id})")
            print(f"  coverage segments: {len(have)}; missing segments: {len(missing)}")

            for ms, me in missing:
                if total_inserted >= max_messages:
                    break

                segment_capped = False

                ms_iso = ms.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                me_iso = me.replace(microsecond=0).isoformat().replace("+00:00", "Z")

                print(f"  syncing missing: {ms_iso} .. {me_iso}")
                seg_total_s = max(1.0, (me - ms).total_seconds())

                def _on_page(meta):
                    last = meta.get("last") or {}
                    last_received = last.get("receivedDateTime")
                    time_pct = None
                    if last_received:
                        try:
                            cur_dt = _parse_iso_utc(str(last_received))
                            time_pct = max(0.0, min(1.0, (cur_dt - ms).total_seconds() / seg_total_s))
                        except Exception:
                            time_pct = None

                    page = str(meta.get("page"))
                    page_part = f"page={_green(page)}"

                    if time_pct is None:
                        extra = f"last_received={last_received}" if last_received else "last_received=?"
                    else:
                        pct = _blue(f"{time_pct * 100.0:.1f}%")
                        extra = f"time_progress={pct} last_received={last_received}"

                    print(
                        f"    {page_part} fetched_total={meta.get('total')} "
                        f"elapsed={meta.get('elapsed_s', 0.0):.1f}s {extra}",
                        flush=True,
                    )

                msgs = graph.iter_messages_in_range(
                    upn,
                    folder_id,
                    start_iso=ms_iso,
                    end_iso=me_iso,
                    page_size=page_size,
                    on_page=_on_page,
                )
                for m in msgs:
                    if total_inserted >= max_messages:
                        segment_capped = True
                        break

                    mid = m.get("id")
                    if not mid:
                        continue

                    body = m.get("body") or {}
                    content = body.get("content") or ""
                    ctype = (body.get("contentType") or "").lower()
                    body_html = None
                    if ctype == "html":
                        body_text = _strip_html_to_text(content)
                        body_html = content
                    else:
                        body_text = content

                    def _recips(key: str):
                        arr = m.get(key) or []
                        out = []
                        for r in arr:
                            ea = (r.get("emailAddress") or {})
                            out.append({
                                "name": ea.get("name") or "",
                                "address": ea.get("address") or "",
                            })
                        return out

                    frm = (m.get("from") or {}).get("emailAddress") or {}

                    row = {
                        "graph_id": str(mid),
                        "folder_id": str(folder_id),
                        "received_dt": m.get("receivedDateTime"),
                        "sent_dt": m.get("sentDateTime"),
                        "last_modified_dt": m.get("lastModifiedDateTime"),
                        "subject": m.get("subject"),
                        "from_name": frm.get("name"),
                        "from_address": frm.get("address"),
                        "to_json": json.dumps(_recips("toRecipients")),
                        "cc_json": json.dumps(_recips("ccRecipients")),
                        "bcc_json": json.dumps(_recips("bccRecipients")),
                        "conversation_id": m.get("conversationId"),
                        "internet_message_id": m.get("internetMessageId"),
                        "body_preview": m.get("bodyPreview"),
                        "body_text": body_text,
                        "body_html": body_html,
                        "has_attachments": 1 if m.get("hasAttachments") else 0,
                    }

                    db.upsert_message(row)

                    if row.get("has_attachments") and attachments_mode in {"full", "ids"}:
                        try:
                            attachments = graph.list_attachments(upn, str(mid))
                        except requests.HTTPError as e:
                            detail = ""
                            try:
                                detail = e.response.text
                            except Exception:
                                pass
                            print(f"  WARN: attachments list failed for message {mid}: {e} {detail}")
                            attachments = []

                        for a in attachments:
                            aid = a.get("id")
                            if not aid:
                                continue

                            atype = str(a.get("@odata.type") or "")
                            name = a.get("name")
                            content_type = a.get("contentType")
                            size = a.get("size")
                            is_inline = 1 if a.get("isInline") else 0
                            content_id = a.get("contentId")
                            last_modified = a.get("lastModifiedDateTime")

                            content_bytes = None
                            content_text = None
                            downloaded = 0
                            err = None

                            if attachments_mode == "full":
                                try:
                                    detail = graph.get_attachment(upn, str(mid), str(aid))
                                    if not content_id:
                                        content_id = detail.get("contentId")
                                    cb = detail.get("contentBytes")
                                    if cb:
                                        content_bytes = base64.b64decode(cb)
                                    else:
                                        # For some attachments Graph omits contentBytes; try $value.
                                        content_bytes = graph.download_attachment_value(upn, str(mid), str(aid))

                                    downloaded = 1 if content_bytes else 0
                                    if content_bytes and _is_text_like_attachment(str(content_type or "")):
                                        content_text = _bytes_to_text_best_effort(content_bytes)
                                    elif content_bytes and _is_pdf_attachment(name, str(content_type or "")):
                                        content_text = _pdf_bytes_to_text_best_effort(content_bytes)
                                except Exception as e:
                                    err = f"download failed: {e}"
                                    downloaded = 0

                            db.upsert_attachment(
                                {
                                    "message_graph_id": str(mid),
                                    "attachment_id": str(aid),
                                    "attachment_type": atype,
                                    "name": name,
                                    "content_type": content_type,
                                    "size": int(size) if size is not None else None,
                                    "is_inline": is_inline,
                                    "content_id": content_id,
                                    "last_modified_dt": last_modified,
                                    "content_bytes": content_bytes,
                                    "content_text": content_text,
                                    "downloaded": downloaded,
                                    "error": err,
                                }
                            )

                    total_inserted += 1

                db.conn.commit()

                # Only mark coverage complete if we did not stop early due to max_messages_per_run.
                if segment_capped:
                    print(
                        "  NOTE: hit max_messages_per_run mid-segment; not updating coverage for this segment. "
                        "Rerun sync to continue."
                    )
                    break

                # Update coverage: add this segment and merge.
                have.append((ms, me))
                merged = _merge_intervals(have)
                db.set_coverage(
                    folder_id,
                    [
                        (
                            s.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                            e.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                        )
                        for s, e in merged
                    ],
                )
                db.conn.commit()

            print(f"  messages upserted so far: {total_inserted}")

        print(f"Done. messages upserted: {total_inserted}")

    finally:
        db.close()


def _mailboxsync_dry(config_path: str, *, sqlite_override: str | None = None):
    cfg = _load_config(config_path)

    tenant_env = cfg.get("graph", {}).get("tenant_id_env")
    client_env = cfg.get("graph", {}).get("client_id_env")
    secret_env = cfg.get("graph", {}).get("client_secret_env")
    upn_env = cfg.get("mailbox", {}).get("mailbox_upn_env")

    if not all([tenant_env, client_env, secret_env, upn_env]):
        raise RuntimeError("Config missing required graph/mailbox env var names")

    tenant_id = _env_required(str(tenant_env))
    client_id = _env_required(str(client_env))
    client_secret = _env_required(str(secret_env))
    upn = _env_required(str(upn_env))

    cache_range = cfg.get("sync", {}).get("cache_range", {})
    start_raw = str(cache_range.get("start_date"))
    end_raw = str(cache_range.get("end_date"))

    if not start_raw or start_raw == "None":
        raise RuntimeError("sync.cache_range.start_date is required")
    if not end_raw or end_raw == "None":
        raise RuntimeError("sync.cache_range.end_date is required")

    start_dt = _parse_iso_utc(start_raw)
    end_dt = _parse_iso_utc(_utc_now_iso() if end_raw.lower() == "now" else end_raw)

    if start_dt >= end_dt:
        raise RuntimeError("cache_range start_date must be < end_date")

    folders_cfg = cfg.get("sync", {}).get("folders", {})
    folder_names = folders_cfg.get("names")
    if not isinstance(folder_names, list) or not folder_names:
        raise RuntimeError("sync.folders.names must be a non-empty list")

    folder_names_norm = [str(x).strip() for x in folder_names if str(x).strip()]

    if sqlite_override:
        sqlite_path = os.path.abspath(sqlite_override)
    else:
        storage = cfg.get("storage", {})
        sqlite_path = str(storage.get("sqlite_path") or "./mail_cache.sqlite")
        sqlite_path = os.path.abspath(os.path.join(os.path.dirname(config_path), sqlite_path))

    print(f"sync dry-run: mailbox={upn}")
    print(f"sync dry-run: cache_range={start_dt.isoformat()}Z..{end_dt.isoformat()}Z")
    print(f"sync dry-run: db={sqlite_path}")

    attachments_mode = (
        (cfg.get("sync", {}).get("attachments", {}) or {}).get("mode")
        or "full"
    )
    attachments_mode = str(attachments_mode).strip().lower()
    if attachments_mode not in {"full", "ids"}:
        raise RuntimeError("sync.attachments.mode must be one of: full, ids")
    print(f"sync dry-run: attachments_mode={attachments_mode}")

    graph = GraphClient(tenant_id, client_id, client_secret)
    folders = graph.list_folders(upn)
    by_id = {f.get("id"): f for f in folders if f.get("id")}

    def folder_path(fid: str) -> str:
        f = by_id.get(fid) or {}
        name = f.get("displayName") or "(unnamed)"
        parent = f.get("parentFolderId")
        if parent and parent in by_id:
            return folder_path(parent) + " / " + name
        return name

    wanted = {n.lower(): n for n in folder_names_norm}
    resolved = []
    for f in folders:
        fid = f.get("id")
        if not fid:
            continue
        p = folder_path(fid)
        dn = (f.get("displayName") or "").strip()
        if dn.lower() in wanted or p.lower() in wanted:
            resolved.append((fid, p))

    if not resolved:
        raise RuntimeError(f"Could not resolve folders from config: {folder_names_norm}")

    db = Db(sqlite_path)
    try:
        db.init_schema()
        for folder_id, folder_p in resolved:
            have = [(_parse_iso_utc(s), _parse_iso_utc(e)) for s, e in db.get_coverage(folder_id)]
            want = (start_dt, end_dt)
            missing = _missing_intervals(want, have)
            print(f"Folder: {folder_p} ({folder_id})")
            print(f"  coverage segments: {len(have)}; missing segments: {len(missing)}")
            for ms, me in missing:
                ms_iso = ms.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                me_iso = me.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                print(f"  would sync: {ms_iso} .. {me_iso}")
    finally:
        db.close()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="sync: local mailbox cache builder")
    ap.add_argument(
        "--config",
        default=os.environ.get("APP_CONFIG") or os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml",
    )
    ap.add_argument(
        "--sqlite",
        default=None,
        help="Override SQLite db path (useful for test runs without editing config)",
    )
    ap.add_argument(
        "cmd",
        nargs="?",
        default="sync",
        choices=["sync", "dry"],
        help="Command to run",
    )

    args = ap.parse_args(argv)

    if args.cmd == "sync":
        _mailboxsync(os.path.abspath(args.config), sqlite_override=args.sqlite)
        return 0

    if args.cmd == "dry":
        _mailboxsync_dry(os.path.abspath(args.config), sqlite_override=args.sqlite)
        return 0

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
