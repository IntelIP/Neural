"""
Dockerfile templates for the Neural SDK deployment module.

This module provides Jinja2-based Dockerfile templates for building
trading bot Docker images with various configurations.
"""

from jinja2 import Template

# Base Dockerfile template for trading bots
DOCKERFILE_TEMPLATE = """FROM python:{{ python_version }}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    postgresql-client \\
    curl \\
    jq \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Create trading user
RUN useradd --create-home --shell /bin/bash trading

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Neural SDK{% if install_neural_sdk %}
RUN pip install --no-cache-dir neural-sdk{% if neural_sdk_version %}=={{ neural_sdk_version }}{% endif %}{% if neural_sdk_extras %}[{{ neural_sdk_extras|join(',') }}]{% endif %}
{% endif %}

# Copy application code
COPY src/ ./src/
{% if include_examples %}COPY examples/ ./examples/
{% endif %}

# Create directories for algorithm injection
# Note: /app/logs removed - use /tmp/monitoring.log for writable logs in sandboxes
RUN mkdir -p /app/algorithms /app/data

{% if entrypoint_script %}# Copy entrypoint script
COPY {{ entrypoint_script }} /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
{% endif %}

# Set ownership
RUN chown -R trading:trading /app

# Switch to non-root user
USER trading

# Environment variables for algorithm configuration
ENV ALGORITHM_TYPE={{ algorithm_type }}
ENV ENVIRONMENT={{ environment }}
ENV BOT_NAME={{ bot_name }}
ENV DATABASE_ENABLED={{ database_enabled|lower }}
ENV WEBSOCKET_ENABLED={{ websocket_enabled|lower }}
ENV MONITORING_ENABLED={{ monitoring_enabled|lower }}

{% if healthcheck_enabled %}# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
{% endif %}

# Default entrypoint
{% if entrypoint_script %}ENTRYPOINT ["/app/entrypoint.sh"]
{% else %}CMD ["python", "-m", "{{ main_module }}"]
{% endif %}
"""

# Multi-stage Dockerfile template (optimized for production)
MULTISTAGE_DOCKERFILE_TEMPLATE = """# Stage 1: Builder
FROM python:{{ python_version }}-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Install Neural SDK
RUN pip install --user --no-cache-dir neural-sdk{% if neural_sdk_version %}=={{ neural_sdk_version }}{% endif %}{% if neural_sdk_extras %}[{{ neural_sdk_extras|join(',') }}]{% endif %}

# Stage 2: Runtime
FROM python:{{ python_version }}-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    postgresql-client \\
    curl \\
    jq \\
    && rm -rf /var/lib/apt/lists/*

# Create trading user
RUN useradd --create-home --shell /bin/bash trading

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/trading/.local

# Copy application code
COPY --chown=trading:trading src/ ./src/
{% if include_examples %}COPY --chown=trading:trading examples/ ./examples/
{% endif %}

# Create directories
RUN mkdir -p /app/algorithms /app/data && chown -R trading:trading /app

{% if entrypoint_script %}# Copy entrypoint
COPY --chown=trading:trading {{ entrypoint_script }} /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
{% endif %}

# Switch to non-root user
USER trading

# Update PATH
ENV PATH=/home/trading/.local/bin:$PATH

# Environment variables
ENV ALGORITHM_TYPE={{ algorithm_type }}
ENV ENVIRONMENT={{ environment }}
ENV BOT_NAME={{ bot_name }}
ENV DATABASE_ENABLED={{ database_enabled|lower }}
ENV WEBSOCKET_ENABLED={{ websocket_enabled|lower }}
ENV MONITORING_ENABLED={{ monitoring_enabled|lower }}

{% if healthcheck_enabled %}# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1
{% endif %}

# Entrypoint
{% if entrypoint_script %}ENTRYPOINT ["/app/entrypoint.sh"]
{% else %}CMD ["python", "-m", "{{ main_module }}"]
{% endif %}
"""

# .dockerignore template
DOCKERIGNORE_TEMPLATE = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.gitignore
.mypy_cache
.pytest_cache
.hypothesis
secrets/
logs/
*.sqlite3
*.db
.env
.env.*
*.md
docs/
tests/
htmlcov/
dist/
build/
*.egg-info/
"""


def render_dockerfile(
    python_version: str = "3.11",
    algorithm_type: str = "mean_reversion",
    environment: str = "sandbox",
    bot_name: str = "Neural-Bot",
    database_enabled: bool = False,
    websocket_enabled: bool = True,
    monitoring_enabled: bool = True,
    install_neural_sdk: bool = True,
    neural_sdk_version: str | None = None,
    neural_sdk_extras: list[str] | None = None,
    include_examples: bool = False,
    entrypoint_script: str | None = None,
    main_module: str = "src.main",
    healthcheck_enabled: bool = False,
    multistage: bool = False,
) -> str:
    """Render a Dockerfile from template.

    Args:
        python_version: Python version (e.g., "3.11", "3.10")
        algorithm_type: Trading algorithm type
        environment: Deployment environment (sandbox/paper/live)
        bot_name: Name of the trading bot
        database_enabled: Enable database persistence
        websocket_enabled: Enable WebSocket trading
        monitoring_enabled: Enable monitoring
        install_neural_sdk: Whether to install neural-sdk package
        neural_sdk_version: Specific version to install (None = latest)
        neural_sdk_extras: Extra dependencies to install (e.g., ["deployment"])
        include_examples: Include examples directory in image
        entrypoint_script: Path to entrypoint script (relative to build context)
        main_module: Main Python module to run (if no entrypoint)
        healthcheck_enabled: Enable Docker healthcheck
        multistage: Use multi-stage build for smaller images

    Returns:
        Rendered Dockerfile as string
    """
    template_str = MULTISTAGE_DOCKERFILE_TEMPLATE if multistage else DOCKERFILE_TEMPLATE

    template = Template(template_str)

    return template.render(
        python_version=python_version,
        algorithm_type=algorithm_type,
        environment=environment,
        bot_name=bot_name,
        database_enabled=database_enabled,
        websocket_enabled=websocket_enabled,
        monitoring_enabled=monitoring_enabled,
        install_neural_sdk=install_neural_sdk,
        neural_sdk_version=neural_sdk_version,
        neural_sdk_extras=neural_sdk_extras or [],
        include_examples=include_examples,
        entrypoint_script=entrypoint_script,
        main_module=main_module,
        healthcheck_enabled=healthcheck_enabled,
    )


def render_dockerignore() -> str:
    """Render a .dockerignore file.

    Returns:
        Rendered .dockerignore content as string
    """
    return DOCKERIGNORE_TEMPLATE.strip()
