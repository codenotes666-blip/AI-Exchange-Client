from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


_DEFAULT_KEYS = (
    "OPENAI_API_KEY",
    "TENANT_ID",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "MAILBOX_UPN",
)


def _candidate_files() -> list[Path]:
    env_override = os.environ.get("AI_EXCHANGE_ENV_FILE")
    if env_override and env_override.strip():
        return [Path(env_override.strip()).expanduser()]

    home = Path.home()
    return [
        home / ".config" / "ai-exchange" / "env",
        home / ".config" / "ai-exchange" / "keys.env",
        home / ".ai-exchange.env",
    ]


def _parse_env_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # allow: export KEY=VALUE
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        out[key] = val
    return out


def _load_from_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return _parse_env_lines(text)


def _load_from_bashrc() -> dict[str, str]:
    # Best-effort fallback for existing setups.
    # Only reads simple export KEY=VALUE lines; does not execute shell.
    p = Path.home() / ".bashrc"
    return _load_from_file(p)


def load_env(*, keys: Iterable[str] = _DEFAULT_KEYS, overwrite: bool = False) -> dict[str, str]:
    """Populate os.environ from a shared per-user env file.

    Priority:
    1) existing os.environ (unless overwrite=True)
    2) AI_EXCHANGE_ENV_FILE or ~/.config/ai-exchange/env (etc.)
    3) ~/.bashrc as a legacy fallback

    Returns a dict of keys that were set.
    """

    wanted = list(keys)
    set_now: dict[str, str] = {}

    # Prefer explicit/standard env files.
    merged: dict[str, str] = {}
    for f in _candidate_files():
        merged.update(_load_from_file(f))

    if not merged:
        merged.update(_load_from_bashrc())

    for k in wanted:
        if not overwrite and os.environ.get(k):
            continue
        if k in merged and merged[k] != "":
            os.environ[k] = merged[k]
            set_now[k] = "***"

    return set_now
