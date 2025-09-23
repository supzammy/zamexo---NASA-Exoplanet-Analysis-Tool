.PHONY: setup dev-setup fmt lint test

PY := python3
PWDQ := $(shell pwd)

setup:
    @$(PY) -m pip install -r requirements.txt

dev-setup:
    @$(PY) -m pip install -r requirements-dev.txt

fmt:
    @ruff check . --select I --fix
    @black -j 1 .

lint:
    @ruff check .
    @black --check -j 1 .

test:
    @PYTHONPATH="$(PWDQ):$$PYTHONPATH" $(PY) -m pytest -q