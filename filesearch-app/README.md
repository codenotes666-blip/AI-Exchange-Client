# filesearch-app

A Windows-only command-line filesystem search agent built on Voidtools **Everything** SDK.

It works like the mailbox cache agent you already have, but instead of searching a SQLite mail cache, it:
- uses the Everything SDK DLL (`Everything64.dll`/`Everything32.dll`) to search the Windows filesystem instantly
- optionally uses OpenAI (your `OPENAI_API_KEY`) to translate natural-language queries into:
  - Everything query syntax (e.g. `ext:txt|md dm:today`, etc.)
  - structured post-filters (e.g. "modified within last N seconds")

## Requirements

- Windows 10/11
- Voidtools Everything installed and running
- Everything SDK DLL available locally
  - Place `Everything64.dll` next to `filesearch_cli.py`, OR
  - Set `EVERYTHING_SDK_DLL` to the absolute path of the DLL
- Python 3.10+
- `requests` (for OpenAI planning)

## Environment

- `OPENAI_API_KEY` (optional, enables planning)
- `EVERYTHING_SDK_DLL` (optional, path to `Everything64.dll` or `Everything32.dll`)
- `AI_EXCHANGE_ENV_FILE` (optional, path to a shared key file)

This CLI will also auto-load missing keys from:
- `~/.config/ai-exchange/env` (recommended), or
- `~/.bashrc` as a legacy fallback.

## Usage

Basic search (no OpenAI planning):

- `python filesearch_cli.py --no-openai "report ext:pdf"`

Natural-language search with planning:

- `python filesearch_cli.py "show me all files modified in the last five minutes that are text files"`

JSON output (for bridging to other tools):

- `python filesearch_cli.py --json --limit 10 "ppt from last week"`

## Notes

- The agent NEVER executes arbitrary commands; it only:
  - calls Everything SDK query functions, and
  - optionally asks OpenAI to produce a JSON plan.
- If Everything can't express something precisely in its query language (e.g. minute-level timestamps depending on SDK/query support), this tool will do **post-filtering in Python** using the returned `date_modified` metadata.
