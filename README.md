# AI Exchange Client

A local, Graph-backed mail client + cache + agent.

This repo is a **monorepo** with three apps:
- `sync-app/`: builds/updates a local SQLite cache from Microsoft Graph (folders + date range + attachments)
- `agent-app/`: intent→safe query templates over the local cache, plus `sync now` / `sync status`
- `web-server-app/`: local 3-pane mail UI + chat grounded in the cache; shells out to `agent-app`

## Prereqs

- Linux + system Python 3.10+
- APT packages (no pip/venv assumed):
  - `python3-msal`
  - `python3-requests`
  - `python3-yaml`
  - Optional for PDF text extraction: `poppler-utils` (provides `pdftotext`)

## Required environment variables

- `TENANT_ID`
- `CLIENT_ID`
- `CLIENT_SECRET`
- `MAILBOX_UPN`
- `OPENAI_API_KEY` (needed for chat; optional for agent planning)

## Configure

1) Copy the example config:

- `cp sync-app/config.example.yaml sync-app/config.yaml`

2) Edit `sync-app/config.yaml`:
- Adjust `sync.cache_range.start_date` / `end_date`
- Confirm `sync.folders.names`
- Leave secrets as env-var references

## Run

### Build/update the cache

- `python3 sync-app/mailboxsync.py --config sync-app/config.yaml sync`

### Use the agent CLI

- `python3 agent-app/cache_agent_cli.py --config sync-app/config.yaml "pdfs from annie"`
- `python3 agent-app/cache_agent_cli.py --config sync-app/config.yaml "sync now"`
- `python3 agent-app/cache_agent_cli.py --config sync-app/config.yaml "sync status"`

### Start the web UI

- `cd web-server-app && bash restart_mail_web_server.sh`

Then open:
- `http://127.0.0.1:8001/`

## Notes

- The web server and the agent both default to `sync-app/config.yaml`.
- The agent emits cache coverage warnings for date-based queries and can kick off a full sync.

## Publishing to GitHub

This workspace initializes a local git repo. To publish as a GitHub repo named **AI Exchange Client**:

- `git add -A`
- `git commit -m "Initial import"`
- Create an empty repo on GitHub named `AI Exchange Client` (or `ai-exchange-client`).
- Add the remote and push:
  - `git remote add origin <YOUR_GITHUB_URL>`
  - `git push -u origin main`
