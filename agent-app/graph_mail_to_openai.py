import os
import requests
import msal
import json

try:
    from env_loader import load_env

    load_env()
except Exception:
    pass

# =========================
# Configuration (ENV ONLY)
# =========================

TENANT_ID = os.environ.get("TENANT_ID")
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
MAILBOX_UPN = os.environ.get("MAILBOX_UPN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

REQUIRED = {
    "TENANT_ID": TENANT_ID,
    "CLIENT_ID": CLIENT_ID,
    "CLIENT_SECRET": CLIENT_SECRET,
    "MAILBOX_UPN": MAILBOX_UPN,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}

missing = [k for k, v in REQUIRED.items() if not v]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# =========================
# Microsoft Graph Auth
# =========================

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def get_graph_token():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=SCOPE)
    if "access_token" not in result:
        raise RuntimeError(f"Graph auth failed: {result}")
    return result["access_token"]


def get_recent_messages(token, limit=5):
    headers = {"Authorization": f"Bearer {token}"}
    url = (
        f"{GRAPH_BASE}/users/{MAILBOX_UPN}/messages"
        f"?$top={limit}"
        f"&$orderby=receivedDateTime desc"
        f"&$select=subject,from,receivedDateTime,bodyPreview"
    )
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["value"]


# =========================
# OpenAI Analysis
# =========================

def analyze_with_openai(message):
    prompt = f"""
Analyze this email:

From: {message['from']['emailAddress']['address']}
Received: {message['receivedDateTime']}
Subject: {message['subject']}

Preview:
{message.get('bodyPreview', '')}

Return:
1. One-sentence summary
2. Any action items
3. Category (finance, legal, aviation, ops, personal, other)
"""

    payload = {
        "model": "gpt-5.1",
        "input": prompt
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if r.status_code >= 400:
        print("\n--- OpenAI error response body ---")
        print(r.text)
        print("--- end OpenAI error ---\n")
        r.raise_for_status()

    out = []
    for item in r.json().get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))

    return "\n".join(out).strip()


# =========================
# Main
# =========================

def main():
    token = get_graph_token()
    messages = get_recent_messages(token)

    print(f"\nRetrieved {len(messages)} messages\n")

    for i, msg in enumerate(messages, 1):
        print("=" * 80)
        print(f"{i}. {msg['subject']}")
        analysis = analyze_with_openai(msg)
        print(analysis)


if __name__ == "__main__":
    main()

