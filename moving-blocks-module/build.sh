#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p dist
tar --exclude='./dist' --exclude='./.venv' --exclude='./.git' \
  --exclude='__pycache__' --exclude='*.pyc' \
  -czf dist/archive.tar.gz .
echo "Wrote dist/archive.tar.gz"
