#!/bin/sh
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python src/main.py "$@"
fi
exec python3 src/main.py "$@"
