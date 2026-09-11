.PHONY: verify lint format test run-api run-ui

verify: lint test

lint:
	ruff check . --fix
	ruff format .

test:
	pytest -q

run-api:
	uvicorn interfaces.api_server:app --reload --port 8000

run-ui:
	streamlit run interfaces/streamlit_app.py
