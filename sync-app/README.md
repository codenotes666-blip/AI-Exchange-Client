# sync-app (mailboxsync)

This folder contains `mailboxsync.py`, the local “OST-like” cache builder.

- Reads config from `config.yaml` (create it from `config.example.yaml`)
- Syncs messages from Microsoft Graph into a local SQLite database
- Attachments are configurable:
	- `sync.attachments.mode: full` stores attachment bytes in SQLite
	- `sync.attachments.mode: ids` stores metadata only
- Intended to support fast local search (message body + recipients + attachment text)

What it caches / indexes:
- Messages:
	- Stores `body_text` and `body_html` (HTML is used by the web UI for correct rendering)
	- Full-text search index: `fts_messages(subject, body_text)` (FTS5)
- Attachments (when `sync.attachments.mode: full`):
	- Stores `content_bytes` and best-effort `content_text`
	- Full-text search index: `fts_attachments(name, content_text)` (FTS5)

Incremental behavior:
- Tracks per-folder coverage intervals in the `coverage` table.
- Re-running sync is idempotent and fills missing time ranges instead of re-downloading everything.

PDF text extraction:
- Text-like attachments (`text/*`, JSON/XML/YAML) are decoded into `attachments.content_text` automatically.
- PDF text extraction is best-effort and requires `pdftotext` (APT: `sudo apt install poppler-utils`).
	- After installing, re-run sync to populate `content_text` for PDFs and update `fts_attachments`.

Key config fields:

- `sync.cache_range.start_date` / `sync.cache_range.end_date` (ISO-8601; `end_date: now` supported)
- `sync.folders.names` (list of folder display names or full paths like `Inbox / Subfolder`)
- `storage.sqlite_path` (relative to config file directory)

See the monorepo overview in [README.md](../README.md).
