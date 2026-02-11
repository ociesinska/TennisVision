.PHONY: format lint format_and_lint

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

format_and_lint: lint format
