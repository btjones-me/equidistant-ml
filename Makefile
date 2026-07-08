.PHONY: install install-dev lint format test clean setup run-server stop-server generate-data generate-data-dry-run

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

generate-data:
	uv run python -m equidistant_ml.etl.get_lattice_data

generate-data-dry-run:
	uv run python -m equidistant_ml.etl.get_lattice_data --dry-run --nrows 5 --output-format csv

# Application commands
run-server:
	uv run uvicorn equidistant_ml.app:app --host 0.0.0.0 --port 8082 --reload

stop-server:
	@echo "Stopping API server..."
	@pkill -f "uvicorn equidistant_ml.app:app" || echo "No server running"

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
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf gmaps_cache.sqlite*
