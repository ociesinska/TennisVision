# Development and testing

[Documentation index](README.md)

Install the locked development dependencies and run the same checks as CI:

```bash
uv sync --locked --extra dev
make check
```

`make check` runs pytest, then Ruff lint and formatting checks without changing files.
Ruff currently checks the video postprocessing tests and the related tracking
modules listed in `CHECK_PATHS` in the Makefile. Pytest runs the complete test suite.
The tests use synthetic detections and require no model weights, videos, GPU, or MLflow server.

Integration tests in `tests/integration/video_detection` exercise the postprocessing
CLI with temporary JSON files, including output artifacts, reloading results, empty
sequences, and invalid input. They use real postprocessing and summary calculations;
MLflow logging and video rendering are replaced with test doubles. Importing the CLI
still requires the project's ML dependencies. Run these tests separately with
`uv run --no-sync python -m pytest tests/integration`.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on pull requests and
pushes to `main`, using Python 3.12 on Ubuntu. It installs the existing project and
development dependencies from `uv.lock`, with uv caching to reduce repeat downloads.
This currently includes the full ML dependency stack; model evaluation is not run.
Check the **Tests and Ruff** job in the pull request checks or repository Actions tab
for results. The check stops at the first failed command; its log shows the failure.

If `UV_PROJECT_ENVIRONMENT` is set locally, uv uses that environment for both commands.
To use `.venv` explicitly, prefix both commands with `UV_PROJECT_ENVIRONMENT=.venv`.

## Adding tests

Keep deterministic calculations separate from model inference where practical.
Place focused tests under `tests/unit/<task>/` and file/pipeline boundary tests under
`tests/integration/<task>/`. Use small synthetic inputs, and place fixtures locally
unless multiple modules actually share them. The current suite focuses on video tracking.

Pytest discovers both suites through `testpaths = ["tests"]` in
[`pyproject.toml`](../pyproject.toml). The Ruff scope is explicit in
[`Makefile`](../Makefile); extend it when adding maintained modules. `make lint` and
`make format` modify files, whereas `make check` only checks them.

ML evaluation is separate from unit-test CI: it uses models and held-out data,
records task metrics in MLflow, and is summarized in the experiment records.

## Documentation conventions

- README introduces the project and links into this directory.
- Shared guides explain setup, architecture, and development.
- Task guides describe current behavior and runnable commands.
- Experiment records preserve run IDs, dataset versions, comparisons, and decisions.

For a new experiment, record the question, dataset/split, configuration change,
training and evaluation run IDs, metrics, qualitative observations, and decision.
Distinguish measured results from hypotheses. Keep one canonical results table per
task and link to it instead of copying it into multiple guides. Keep large datasets,
weights, and videos in their existing local/artifact stores.
