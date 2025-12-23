# web-server-app

Local 3-pane mail UI (folders → list → message) plus a chat box grounded in the local SQLite cache.

The web server shells out to the external agent:
- `../agent-app/cache_agent_cli.py --json ...`

## Run

- `cd web-server-app && bash restart_mail_web_server.sh`

Then open:
- `http://127.0.0.1:8001/`

## Config

By default the server reads:
- `../sync-app/config.yaml`

Create it from the example:
- `cp ../sync-app/config.example.yaml ../sync-app/config.yaml`

## Endpoints

- `POST /chat` – chat grounded in cache results
- `POST /agent/search` – returns raw agent JSON
- `GET /sync/status` – server-side sync status (auto-sync system)
