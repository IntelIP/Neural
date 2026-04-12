.PHONY: help install install-dev lint type test clean build check release-dry-run publish-testpypi publish audit audit-security audit-deps
.DEFAULT_GOAL := help

PYTHON ?= python3
USER_SITE := $(shell $(PYTHON) -c "import site; print(site.getusersitepackages())")

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

audit: ## Run quality gates for bug analysis
	ruff check neural tests scripts utils
	mypy neural
	pytest tests/

audit-security: ## Run static security scan (Bandit)
	bandit -q -r neural

audit-deps: ## Run dependency vulnerability scan
	pip-audit -r requirements-dev.txt

clean: ## Clean build artifacts
	rm -rf build dist neural.egg-info neural_sdk.egg-info *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	$(PYTHON) -m build --no-isolation

check: build ## Check built package with twine
	twine check dist/*

release-dry-run: build check ## Validate release artifacts without publishing
	$(PYTHON) -m venv --system-site-packages /tmp/neural-release-wheel-venv
	/tmp/neural-release-wheel-venv/bin/python -m pip install dist/*.whl
	cd /tmp && /tmp/neural-release-wheel-venv/bin/python "$(CURDIR)/scripts/package_smoke.py"
	$(PYTHON) -m venv --system-site-packages /tmp/neural-release-sdist-venv
	PYTHONPATH="$(USER_SITE):$$PYTHONPATH" /tmp/neural-release-sdist-venv/bin/python -m pip install --no-build-isolation dist/*.tar.gz
	cd /tmp && PYTHONPATH="$(USER_SITE):$$PYTHONPATH" /tmp/neural-release-sdist-venv/bin/python "$(CURDIR)/scripts/package_smoke.py"

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
