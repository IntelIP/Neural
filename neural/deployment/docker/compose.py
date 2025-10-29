"""
Docker Compose orchestration for the Neural SDK deployment module.

This module provides utilities for generating and managing Docker Compose
configurations for multi-service trading bot deployments.
"""

from pathlib import Path
from typing import Any

from jinja2 import Template

# Docker Compose template for trading bot stack
COMPOSE_TEMPLATE = """version: '3.8'

services:
  trading-bot:
    build: {{ build_context }}
    container_name: {{ container_name }}
    environment:
      - ALGORITHM_TYPE={{ algorithm_type }}
      - ENVIRONMENT={{ environment }}
      - BOT_NAME={{ bot_name }}
      - KALSHI_API_KEY_ID=${KALSHI_API_KEY_ID}
      - KALSHI_PRIVATE_KEY_PATH=/secrets/private_key.pem
      {% if database_enabled %}- DB_HOST=postgres
      - DB_PORT=5432
      - DB_USER=trading_user
      - DB_PASSWORD=trading_pass
      - DB_NAME=trading_db{% endif %}
    volumes:
      - ./secrets:/secrets:ro
      - {{ bot_name|lower|replace(' ', '-') }}_logs:/tmp
    {% if database_enabled %}depends_on:
      - postgres{% endif %}
    networks:
      - trading-network
    restart: unless-stopped
    {% if cpu_limit or memory_limit %}deploy:
      resources:
        limits:
          {% if cpu_limit %}cpus: '{{ cpu_limit }}'{% endif %}
          {% if memory_limit %}memory: {{ memory_limit }}{% endif %}
    {% endif %}

{% if database_enabled %}  postgres:
    image: postgres:15-alpine
    container_name: {{ bot_name|lower|replace(' ', '-') }}-postgres
    environment:
      - POSTGRES_DB=trading_db
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD=trading_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - trading-network
    restart: unless-stopped
{% endif %}

{% if monitoring_enabled %}  prometheus:
    image: prom/prometheus:latest
    container_name: {{ bot_name|lower|replace(' ', '-') }}-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - trading-network
    restart: unless-stopped
{% endif %}

volumes:
  {{ bot_name|lower|replace(' ', '-') }}_logs:
  {% if database_enabled %}postgres_data:{% endif %}
  {% if monitoring_enabled %}prometheus_data:{% endif %}

networks:
  trading-network:
    driver: bridge
"""


def render_compose_file(
    bot_name: str,
    algorithm_type: str = "mean_reversion",
    environment: str = "sandbox",
    build_context: str = ".",
    container_name: str | None = None,
    database_enabled: bool = True,
    monitoring_enabled: bool = False,
    cpu_limit: str | None = None,
    memory_limit: str | None = None,
) -> str:
    """Render a Docker Compose file from template.

    Args:
        bot_name: Name of the trading bot
        algorithm_type: Trading algorithm type
        environment: Deployment environment
        build_context: Docker build context path
        container_name: Custom container name (auto-generated if None)
        database_enabled: Include PostgreSQL service
        monitoring_enabled: Include Prometheus monitoring
        cpu_limit: CPU limit (e.g., "1.0")
        memory_limit: Memory limit (e.g., "2g")

    Returns:
        Rendered docker-compose.yml as string
    """
    if not container_name:
        container_name = f"{bot_name.lower().replace(' ', '-')}-trading-bot"

    template = Template(COMPOSE_TEMPLATE)

    return template.render(
        bot_name=bot_name,
        algorithm_type=algorithm_type,
        environment=environment,
        build_context=build_context,
        container_name=container_name,
        database_enabled=database_enabled,
        monitoring_enabled=monitoring_enabled,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
    )


def write_compose_file(
    output_path: Path,
    bot_name: str,
    **kwargs: Any,
) -> Path:
    """Generate and write a Docker Compose file.

    Args:
        output_path: Path where to write the compose file
        bot_name: Name of the trading bot
        **kwargs: Additional arguments passed to render_compose_file()

    Returns:
        Path to the written compose file
    """
    compose_content = render_compose_file(bot_name=bot_name, **kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(compose_content)

    return output_path
