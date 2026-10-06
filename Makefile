.PHONY: verify lint format test run-api run-ui \
        pm2-start pm2-stop pm2-restart pm2-logs pm2-status pm2-save

verify: lint test

lint:
	ruff check . --fix
	ruff format .

test:
	pytest -q

run-api:
	# --reload is intentionally omitted: harness/ file writes (AGENTS.md, sessions/)
	# trigger uvicorn restart loops. Use 'make pm2-restart' to hot-reload instead.
	uvicorn interfaces.api_server:app --port 8000

run-ui:
	streamlit run interfaces/streamlit_app.py

# ── PM2 background process management ─────────────────────────────────────────
pm2-start:
	pm2 start ecosystem.config.js

pm2-stop:
	pm2 stop api-server

pm2-restart:
	pm2 restart ecosystem.config.js

pm2-logs:
	pm2 logs api-server --lines 100

pm2-status:
	pm2 list

pm2-save:
	pm2 save
