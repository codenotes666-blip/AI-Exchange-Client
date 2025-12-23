# agent-app

This is the **search agent** CLI that:
- Interprets natural-language intent into safe, predefined cache-query templates (no arbitrary SQL)
- Emits JSON for the web server bridge
- Can start and report **full sync** runs (`sync now`, `sync status`)

## Run

- `python3 cache_agent_cli.py --config ../sync-app/config.yaml "pdfs from annie"`

REPL:
- `python3 cache_agent_cli.py --config ../sync-app/config.yaml --repl --explain`

Sync control:
- `python3 cache_agent_cli.py --config ../sync-app/config.yaml "sync now"`
- `python3 cache_agent_cli.py --config ../sync-app/config.yaml "sync status"`

## Notes

- The agent reads the SQLite path from `storage.sqlite_path` in the config.
- If `OPENAI_API_KEY` is set, it can ask the model to pick a strategy; otherwise it uses deterministic fallbacks.
