PY := .venv/bin/python

.PHONY: setup api api-dev ui test lint clean

setup:  ## create .venv and install everything
	bash scripts/setup.sh

api:  ## start the FastAPI backend (host/port from .env)
	$(PY) -m backend.app

api-dev:  ## backend with auto-reload
	$(PY) -m uvicorn backend.app.main:create_app --factory --reload

ui:  ## start the Streamlit front end
	$(PY) -m streamlit run frontend/app.py

test:  ## run the test suite
	$(PY) -m pytest -q

lint:  ## static checks
	$(PY) -m ruff check backend frontend tests

clean:  ## remove caches (keeps your indexes)
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
