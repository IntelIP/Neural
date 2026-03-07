.PHONY: help install install-dev lint type test clean build publish-testpypi audit audit-security audit-deps
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package
	uv sync

install-dev: ## Install package with dev dependencies
	uv sync --extra dev --extra trading --extra sentiment --extra analysis --extra deployment

lint: ## Run linters
	uv run ruff check neural tests
	uv run ruff format --check neural tests

format: ## Format code
	uv run ruff format neural tests

type: ## Run type checker
	uv run mypy neural

test: ## Run tests
	uv run pytest tests/

test-cov: ## Run tests with coverage
	uv run pytest tests/ --cov=neural --cov-report=term-missing

audit: ## Run quality gates for bug analysis
	uv run ruff check neural tests scripts utils
	uv run mypy neural
	uv run pytest tests/

audit-security: ## Run static security scan (Bandit)
	uv tool run bandit -q -r neural

audit-deps: ## Run dependency vulnerability scan
	uv tool run pip-audit -r requirements-dev.txt

clean: ## Clean build artifacts
	rm -rf build dist neural.egg-info neural_sdk.egg-info *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	uv build

check: build ## Check built package with twine
	uv tool run twine check dist/*

publish-testpypi: build check ## Publish to TestPyPI
	uv tool run twine upload --repository testpypi --non-interactive --skip-existing dist/*

publish: build check ## Publish to PyPI
	uv tool run twine upload --non-interactive --skip-existing dist/*

bump-patch: ## Bump patch version (0.1.0 -> 0.1.1)
	bump2version patch

bump-minor: ## Bump minor version (0.1.0 -> 0.2.0)
	bump2version minor

bump-major: ## Bump major version (0.1.0 -> 1.0.0)
	bump2version major
