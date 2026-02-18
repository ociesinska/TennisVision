.PHONY: format lint format_and_lint install 

install:
	uv pip install -e .

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

format_and_lint: lint format


.PHONY: mlflow
mlflow:
	uv run mlflow ui \
		--backend-store-uri ./mlruns \
		--artifacts-destination ./artifacts/mlflow/artifacts \
		--serve-artifacts \
		--port 8080