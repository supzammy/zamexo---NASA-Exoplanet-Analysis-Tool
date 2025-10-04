.PHONY: setup dev-setup fmt lint test run clean all app list-legacy evaluate-models train-baseline train-multisource verify-bls verify-data-sources run-% loc

PY := python3
PWDQ := $(shell pwd)

setup:
	@$(PY) -m pip install -r requirements.txt

dev-setup:
	@$(PY) -m pip install -r requirements-dev.txt

fmt:
	@$(PY) -m black .

lint:
	@ruff check --fix .

test:
	@PYTHONPATH="$(PWDQ):$$PYTHONPATH" $(PY) -m pytest -q

# Streamlit (prefer 01.py as main UI)
app:
	PYTHONPATH="$(pwd):$$PYTHONPATH" streamlit run "app/streamlit_app.py"

# Legacy listing (optional if you keep CLI)
list-legacy:
	@PYTHONPATH="$(PWDQ):$$PYTHONPATH" $(PY) -m nasa_project list || true

# Convenience legacy runners (stubs included)
evaluate-models:
	@$(PY) scripts/evaluate_models.py

train-baseline:
	@$(PY) scripts/train_baseline.py --offline

train-multisource:
	@$(PY) scripts/train_multisource.py --epochs 1

verify-bls:
	@$(PY) scripts/verify_bls.py

verify-data-sources:
	@$(PY) scripts/verify_data_sources.py

train:
	python3 -c 'from scripts.train_baseline import main; print("train accuracy:", main(force_download=False, min_accuracy=0.0))'

prefetch:
	@echo 'Example: make prefetch M=Kepler T="Kepler-10 Kepler-22"'
	@echo 'Usage: make prefetch M=<Kepler|K2|TESS> T="<target1 target2 ...>"'
	python3 -m scripts.prefetch_lc $(M) $(T)

# Pattern: make run-<name>
run-%:
    @echo "Running: $*"; \
    PYTHONPATH="$(PWDQ):$$PYTHONPATH" $(PY) -m nasa_project run $* -- --help || true

# Back-compat
run:
    streamlit run app/streamlit_app.py

clean:
    @find . -name "__pycache__" -type d -prune -exec rm -rf {} +; \
    rm -rf .ruff_cache .pytest_cache .coverage coverage.xml models/*

all: fmt lint test

.PHONY: loc
loc:
    @git ls-files '*.py' | xargs wc -l; echo "----"; cloc app "01.py" || true

cache:
    streamlit cache clear

.PHONY: run-mvp
run-mvp:
	PYTHONPATH="$(PWD):$$PYTHONPATH" streamlit run app/app_mvp.py