.PHONY: install install-dev lint format test check clean setup run-server stop-server generate-data generate-data-dry-run dvc-repro-smoke dvc-repro-graph-smoke generate-traveltime-data fetch-transport-data build-transport-graph graph-hillclimb train train-graph-model evaluate evaluate-corridors export-browser-atlas frontend-install frontend-dev frontend-test frontend-build

# Development commands
install:
	uv sync

install-dev:
	uv sync --extra dev

lint:
	uv run --extra dev flake8 equidistant_ml tests
	uv run --extra dev isort --check-only --profile black equidistant_ml tests
	uv run --extra dev black --check --diff equidistant_ml tests
	uv run --extra dev mypy --ignore-missing-imports equidistant_ml tests
	uv run --extra dev bandit -r equidistant_ml

format:
	uv run --extra dev isort --profile black equidistant_ml tests
	uv run --extra dev black equidistant_ml tests

test:
	uv run --extra dev pytest --verbose --capture=no

check: lint test frontend-test frontend-build

generate-data:
	uv run python -m equidistant_ml.etl.get_lattice_data

generate-data-dry-run:
	uv run python -m equidistant_ml.etl.get_lattice_data --dry-run --nrows 5 --output-format csv

dvc-repro-smoke:
	uv run python -m equidistant_ml.surfaces.pipeline smoke --output-dir data/smoke --max-origins 6 --max-destinations 36
	uv run python -m equidistant_ml.surfaces.pipeline train-baseline --features data/smoke/features.parquet --output data/smoke/baseline_model.joblib
	uv run python -m equidistant_ml.surfaces.pipeline train-model --features data/smoke/features.parquet --output data/smoke/travel_time_model.joblib
	uv run python -m equidistant_ml.surfaces.pipeline evaluate --features data/smoke/features.parquet --baseline-model data/smoke/baseline_model.joblib --model data/smoke/travel_time_model.joblib --output data/smoke/model.json

dvc-repro-graph-smoke:
	uv run python -m equidistant_ml.surfaces.pipeline graph-smoke --output-dir data/smoke_graph --max-origins 6 --max-destinations 36

generate-traveltime-data:
	uv run dvc repro fetch_traveltime

fetch-transport-data:
	uv run dvc repro fetch_transport_reference

build-transport-graph:
	uv run dvc repro build_transport_graph build_graph_features

graph-hillclimb:
	uv run python -m equidistant_ml.surfaces.hillclimb

train:
	uv run dvc repro train_baseline train_model train_graph_baseline train_graph_model

train-graph-model:
	uv run dvc repro train_graph_baseline train_graph_model

evaluate:
	uv run dvc repro evaluate

evaluate-corridors:
	uv run dvc repro evaluate_corridors

export-browser-atlas:
	uv run python -m equidistant_ml.surfaces.export_atlas

# Application commands
run-server:
	uv run uvicorn equidistant_ml.app:app --host 0.0.0.0 --port 8082 --reload

stop-server:
	@echo "Stopping API server..."
	@pkill -f "uvicorn equidistant_ml.app:app" || echo "No server running"

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

frontend-test:
	cd frontend && npm test

frontend-build:
	cd frontend && npm run build

# Setup commands
setup: install-dev
	@echo "Setting up equidistant-ml..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from template"; fi
	@echo "Setup complete. Add any required local configuration to .env."

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf data/smoke
	rm -rf data/experiments
	rm -rf data/smoke_graph
	rm -rf build
	rm -rf dist
	rm -rf frontend/dist
	rm -rf *.egg-info
	rm -rf gmaps_cache.sqlite*
