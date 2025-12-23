#!/usr/bin/env python3
"""filesearch-app CLI (Windows).

Natural language -> (Everything query syntax + safe post-filters) -> Everything SDK search.

This mirrors the style of the mailbox cache agent:
- optional OpenAI planning using OPENAI_API_KEY
- deterministic fallback planning when key isn't present
- JSON output mode for bridging

This tool does NOT execute arbitrary commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from typing import Any

import requests

try:
    from env_loader import load_env

    load_env()
except Exception:
    pass

from everything_sdk import EverythingClient, EverythingSdkError, resolve_dll_path


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _openai_key() -> str | None:
    k = os.environ.get("OPENAI_API_KEY")
    return k.strip() if isinstance(k, str) and k.strip() else None


def _chat_model(override: str | None) -> str:
    return (override or "gpt-5.1").strip() or "gpt-5.1"


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _extract_within_seconds_hint(q: str) -> int | None:
    s = (q or "").lower()

    # "last five minutes" etc.
    m = re.search(r"\b(last|past)\s+(\d{1,4})\s*(seconds?|minutes?|hours?|days?)\b", s)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        if unit.startswith("second"):
            return max(0, min(7 * 24 * 3600, n))
        if unit.startswith("minute"):
            return max(0, min(7 * 24 * 3600, n * 60))
        if unit.startswith("hour"):
            return max(0, min(7 * 24 * 3600, n * 3600))
        if unit.startswith("day"):
            return max(0, min(7 * 24 * 3600, n * 86400))

    # common phrases
    if "last minute" in s:
        return 60
    if "last hour" in s or "past hour" in s:
        return 3600
    if "last 24 hours" in s:
        return 86400

    return None


def _text_file_extensions_hint(q: str) -> list[str] | None:
    s = (q or "").lower()
    if "text file" in s or "text files" in s:
        # Conservative set. User can still ask for a different set.
        return [
            "txt",
            "md",
            "rst",
            "log",
            "csv",
            "tsv",
            "json",
            "yaml",
            "yml",
            "xml",
            "ini",
            "cfg",
            "conf",
            "py",
            "js",
            "ts",
            "html",
            "css",
            "sh",
            "bat",
            "ps1",
        ]
    return None


def _extensions_to_everything_syntax(exts: list[str]) -> str:
    cleaned = []
    for e in exts:
        e2 = (e or "").strip().lstrip(".")
        if not e2:
            continue
        if re.fullmatch(r"[A-Za-z0-9]{1,12}", e2):
            cleaned.append(e2.lower())
    cleaned = cleaned[:30]
    if not cleaned:
        return ""
    # Everything supports ext:foo and OR with | inside the value.
    return "ext:" + "|".join(cleaned)


def _openai_plan(*, model: str, message: str) -> dict:
    key = _openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    system = (
        "You translate natural language into an Everything filesystem search plan. "
        "Return ONLY strict JSON (no markdown). "
        "Do NOT include any code."
    )

    allowed = {
        "strategy": ["everything_search"],
        "fields": {
            "everything_query": "string (Everything query syntax)",
            "filters": {
                "modified_within_seconds": "int|null (optional)",
                "extensions": "list[str]|null (optional)",
                "files_only": "bool|null (optional)",
                "folders_only": "bool|null (optional)",
            },
            "limit": "int (1-10000)",
            "rationale": "short string",
        },
    }

    user = {
        "query": message,
        "allowed": allowed,
        "notes": [
            "Use Everything query syntax for filetype where possible (e.g. ext:txt|md).",
            "For time windows like 'last five minutes', set filters.modified_within_seconds=300.",
            "If you can't express a constraint reliably in Everything syntax, leave it to filters.",
            "Prefer a simple, safe plan.",
        ],
        "output_format": {
            "strategy": "everything_search",
            "everything_query": "...",
            "filters": {
                "modified_within_seconds": None,
                "extensions": None,
                "files_only": None,
                "folders_only": None,
            },
            "limit": 50,
            "rationale": "...",
        },
    }

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:600]}")

    data = r.json() if r.content else {}
    out_texts: list[str] = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                out_texts.append(c.get("text", ""))
    text = "\n".join(out_texts).strip()
    if not text:
        raise RuntimeError("OpenAI returned empty output")

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if not m:
            raise RuntimeError(f"Could not parse JSON from model output: {text[:400]}")
        return json.loads(m.group(0))


def _fallback_plan(message: str, *, limit: int) -> dict:
    within = _extract_within_seconds_hint(message)
    exts = _text_file_extensions_hint(message)

    q_parts: list[str] = []
    if exts:
        q_parts.append(_extensions_to_everything_syntax(exts))

    # Keep the raw message as a keyword search term if we have nothing else.
    # This is imperfect but safe.
    if not q_parts:
        q_parts.append(message.strip())

    return {
        "strategy": "everything_search",
        "everything_query": " ".join([p for p in q_parts if p]).strip(),
        "filters": {
            "modified_within_seconds": within,
            "extensions": exts,
            "files_only": True if ("file" in (message or "").lower()) else None,
            "folders_only": True if ("folder" in (message or "").lower() or "directory" in (message or "").lower()) else None,
        },
        "limit": int(limit),
        "rationale": "fallback (no OpenAI planning)",
    }


def _validate_plan(plan: dict, *, default_limit: int) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")

    strategy = str(plan.get("strategy") or "").strip()
    if strategy != "everything_search":
        raise ValueError(f"Unsupported strategy: {strategy!r}")

    everything_query = str(plan.get("everything_query") or "").strip()
    if not everything_query:
        raise ValueError("everything_query is required")

    limit = _safe_int(plan.get("limit"), default_limit)
    limit = max(1, min(10000, limit))

    filters = plan.get("filters")
    if not isinstance(filters, dict):
        filters = {}

    mws = filters.get("modified_within_seconds")
    if mws is not None:
        mws = _safe_int(mws, 0)
        mws = max(0, min(365 * 24 * 3600, mws))
    exts = filters.get("extensions")
    if exts is not None and not isinstance(exts, list):
        exts = None
    if isinstance(exts, list):
        exts = [str(e) for e in exts if str(e).strip()][:50]

    def _bool_or_none(v):
        if v is None:
            return None
        return bool(v)

    out = {
        "strategy": "everything_search",
        "everything_query": everything_query,
        "filters": {
            "modified_within_seconds": mws,
            "extensions": exts,
            "files_only": _bool_or_none(filters.get("files_only")),
            "folders_only": _bool_or_none(filters.get("folders_only")),
        },
        "limit": limit,
        "rationale": str(plan.get("rationale") or "").strip()[:400],
    }

    # Avoid contradictory flags.
    if out["filters"]["files_only"] and out["filters"]["folders_only"]:
        out["filters"]["folders_only"] = None

    return out


def _apply_post_filters(rows: list[dict], *, plan_filters: dict) -> list[dict]:
    out = rows

    files_only = plan_filters.get("files_only")
    folders_only = plan_filters.get("folders_only")
    if folders_only is True:
        out = [r for r in out if r.get("is_folder") is True]
    elif files_only is True:
        out = [r for r in out if r.get("is_folder") is False]

    within = plan_filters.get("modified_within_seconds")
    if isinstance(within, int) and within > 0:
        cutoff = _utc_now() - dt.timedelta(seconds=within)
        tmp = []
        for r in out:
            dm = r.get("date_modified_utc")
            if isinstance(dm, dt.datetime) and dm >= cutoff:
                tmp.append(r)
        out = tmp

    exts = plan_filters.get("extensions")
    if isinstance(exts, list) and exts:
        norm = {str(e).strip().lstrip(".").lower() for e in exts if str(e).strip()}
        if norm:
            tmp = []
            for r in out:
                p = str(r.get("full_path") or "")
                _, ext = os.path.splitext(p)
                ext = ext.lstrip(".").lower()
                if ext and ext in norm:
                    tmp.append(r)
            out = tmp

    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="filesearch-app CLI (Windows)")
    ap.add_argument("query", nargs="*", help="Natural language or Everything query")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", default=None, help="OpenAI model override")
    ap.add_argument("--no-openai", action="store_true", help="Disable OpenAI planning")
    ap.add_argument("--dll-path", default=None, help="Path to Everything64.dll/Everything32.dll")
    ap.add_argument("--match-case", action="store_true")
    ap.add_argument("--whole-word", action="store_true")
    ap.add_argument("--regex", action="store_true")
    ap.add_argument("--explain", action="store_true", help="Print plan (non-JSON mode)")

    ns = ap.parse_args(argv)

    q = " ".join(ns.query).strip() if ns.query else ""
    if not q:
        q = sys.stdin.read().strip()
    if not q:
        raise RuntimeError("Empty query")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = resolve_dll_path(ns.dll_path, base_dir=base_dir)

    used_openai = False
    raw_plan: dict
    if (not ns.no_openai) and _openai_key():
        try:
            raw_plan = _openai_plan(model=_chat_model(ns.model), message=q)
            used_openai = True
        except Exception as e:
            raw_plan = _fallback_plan(q, limit=ns.limit)
            raw_plan["rationale"] = f"openai failed: {e}"
    else:
        raw_plan = _fallback_plan(q, limit=ns.limit)

    plan = _validate_plan(raw_plan, default_limit=ns.limit)

    if ns.explain and (not ns.json):
        print("== PLAN ==")
        print(json.dumps({"openai_planner": used_openai, **plan}, indent=2, default=str))

    try:
        client = EverythingClient(dll_path)
        results = list(
            client.query(
                search=plan["everything_query"],
                limit=plan["limit"],
                match_case=ns.match_case,
                whole_word=ns.whole_word,
                regex=ns.regex,
                request_size=True,
                request_date_modified=True,
                request_attributes=True,
            )
        )
    except EverythingSdkError as e:
        payload = {"ok": False, "error": str(e), "query": q, "plan": {"openai_planner": used_openai, **plan}}
        if ns.json:
            print(json.dumps(payload, ensure_ascii=False))
            return 2
        raise

    rows: list[dict] = []
    for r in results:
        rows.append(
            {
                "full_path": r.full_path,
                "file_name": r.file_name,
                "path": r.path,
                "size": r.size,
                "date_modified_utc": r.date_modified_utc,
                "is_folder": r.is_folder,
            }
        )

    filtered = _apply_post_filters(rows, plan_filters=plan["filters"])

    payload = {
        "ok": True,
        "query": q,
        "plan": {"openai_planner": used_openai, **plan},
        "results": [
            {
                **r,
                "date_modified_utc": (r.get("date_modified_utc").isoformat().replace("+00:00", "Z") if isinstance(r.get("date_modified_utc"), dt.datetime) else None),
            }
            for r in filtered
        ],
        "count": len(filtered),
    }

    if ns.json:
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(f"== Everything results ({len(filtered)}) ==")
    for i, r in enumerate(payload["results"], 1):
        dm = r.get("date_modified_utc") or ""
        sz = r.get("size")
        sz_s = f" ({sz} bytes)" if isinstance(sz, int) else ""
        print(f"{i}. {r.get('full_path','')}{sz_s}")
        if dm:
            print(f"   modified: {dm}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
