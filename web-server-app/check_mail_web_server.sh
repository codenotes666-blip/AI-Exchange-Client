#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 - <<'PY'
import py_compile
import sys
path = 'mail_web_server.py'
try:
    py_compile.compile(path, doraise=True)
    print('OK: py_compile passed for', path)
except Exception as e:
    print('FAIL: py_compile failed for', path)
    print(e)
    sys.exit(1)
PY
