.PHONY: setup dev-setup test train-baseline train-multi run clean test-cov fmt lint

PY=python3
PWDQ:=$(shell pwd)

setup:
	"$(PY)" -m pip install -r requirements.txt

dev-setup:
	"$(PY)" -m pip install -r requirements-dev.txt

test:
    @PYTHONPATH=$(PWD):$$PYTHONPATH python3 -m pytest -q

# Coverage via coverage.py (works even if pytest-cov flags aren’t registered)
test-cov:
	PYTHONPATH="$(PWDQ):$$PYTHONPATH" "$(PY)" -m coverage run -m pytest -q --maxfail=1
	"$(PY)" -m coverage report -m

train-baseline:
	PYTHONPATH="$(PWDQ):$$PYTHONPATH" "$(PY)" -m scripts.train_baseline

train-multi:
	PYTHONPATH="$(PWDQ):$$PYTHONPATH" "$(PY)" -m scripts.train_multisource --sources ALL

run:
	PYTHONPATH="$(PWDQ):$$PYTHONPATH" streamlit run 01.py

lint:
    @ruff check .
    @black --check .

fmt:
    @ruff check . --select I --fix
    @black .