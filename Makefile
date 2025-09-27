.PHONY: help install install-dev lint type test clean build publish-testpypi
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package
	pip install -e .

install-dev: ## Install package with dev dependencies
	pip install -e ".[dev,trading]"

lint: ## Run linters
	ruff check neural tests
	ruff format --check neural tests

format: ## Format code
	ruff format neural tests

type: ## Run type checker
	mypy neural

test: ## Run tests
	pytest tests/

test-cov: ## Run tests with coverage
	pytest tests/ --cov=neural --cov-report=term-missing

clean: ## Clean build artifacts
	rm -rf build dist neural.egg-info neural_sdk.egg-info *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

check: build ## Check built package with twine
	twine check dist/*

publish-testpypi: build check ## Publish to TestPyPI
	twine upload --repository testpypi --non-interactive --skip-existing dist/*

publish: build check ## Publish to PyPI
	twine upload --non-interactive --skip-existing dist/*

bump-patch: ## Bump patch version (0.1.0 -> 0.1.1)
	bump2version patch

bump-minor: ## Bump minor version (0.1.0 -> 0.2.0)
	bump2version minor

bump-major: ## Bump major version (0.1.0 -> 1.0.0)
	bump2version major