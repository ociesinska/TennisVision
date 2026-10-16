.PHONY: format lint format_and_lint install sync check

CHECK_PATHS = tests/unit/video_detection \
	src/tennisvision/tasks/video_detection/postprocessing.py \
	src/tennisvision/tasks/video_detection/tracking.py \
	src/tennisvision/tasks/video_detection/types.py \
	src/tennisvision/tasks/video_detection/backends/ultralytics_video.py

install: sync

sync:
	uv sync

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

format_and_lint: lint format

# Requires development dependencies: uv sync --extra dev
# Use the installed environment without syncing or modifying source files.
check:
	uv run --no-sync python -m pytest
	uv run --no-sync ruff check $(CHECK_PATHS)
	uv run --no-sync ruff format --check $(CHECK_PATHS)


.PHONY: mlflow
mlflow:
	uv run mlflow ui \
		--backend-store-uri ./mlruns \
		--artifacts-destination ./artifacts/mlflow/artifacts \
		--serve-artifacts \
		--port 8080
