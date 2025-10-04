#!/usr/bin/env bash
set -e
echo "Python: $(python3 -V)"
echo
echo "Ruff (lint) summary:"
ruff check . --count --statistics || true
echo
echo "Tests:"
pytest -q || true
echo
echo "Git status:"
git status -s || true
echo
echo "Diff stat:"
git diff --stat || true
