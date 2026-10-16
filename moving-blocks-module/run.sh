#!/bin/sh
set -e
cd "$(dirname "$0")"

# reload-local builds on the developer machine and ships source only.
# Create the virtualenv here so the Linux robot gets its own wheels.
if [ ! -x .venv/bin/python ]; then
  echo "moving-blocks: creating virtualenv and installing requirements"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python src/main.py "$@"
