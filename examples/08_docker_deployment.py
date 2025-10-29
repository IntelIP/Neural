"""
Example: Deploy a Trading Bot with Docker

This example demonstrates how to use the Neural SDK deployment module
to deploy trading bots in Docker containers.

⚠️ EXPERIMENTAL: The deployment module is experimental in v0.4.0.

Requirements:
    - Docker installed and running
    - neural-sdk[deployment] installed
    - Kalshi API credentials (optional for paper trading)

Usage:
    python examples/08_docker_deployment.py
"""

import asyncio
import os

from neural.deployment import (
    DeploymentConfig,
    DockerDeploymentProvider,
    deploy,
)


async def basic_deployment_example():
    """Basic deployment example using Docker provider."""
    print("=" * 60)
    print("Neural SDK Deployment Module - Basic Example")
    print("=" * 60)
    print()

    # Create deployment configuration
    config = DeploymentConfig(
        bot_name="NFL-MeanReversion-Bot",
        strategy_type="NFL",
        environment="paper",  # Use paper trading mode
        algorithm_config={
            "algorithm_type": "mean_reversion",
            "poll_interval": 30.0,
            "market_limit": 12,
        },
        risk_config={
            "max_position_size": 100,
            "max_total_exposure": 1000,
        },
        database_enabled=False,  # Disable database for this example
        websocket_enabled=True,
        monitoring_enabled=True,
    )

    print(f"📋 Configuration:")
    print(f"   Bot Name: {config.bot_name}")
    print(f"   Strategy: {config.strategy_type}")
    print(f"   Environment: {config.environment}")
    print()

    # Create Docker deployment provider
    provider = DockerDeploymentProvider()

    # Deploy with context manager (auto-cleanup)
    print("🚀 Deploying trading bot...")
    async with deploy(provider, config) as deployment:
        print(f"✅ Deployed successfully!")
        print(f"   Deployment ID: {deployment.deployment_id}")
        print(f"   Container ID: {deployment.container_id[:12]}...")
        print(f"   Container Name: {deployment.container_name}")
        print()

        # Wait a moment for container to start
        await asyncio.sleep(3)

        # Check status
        print("📊 Checking deployment status...")
        status = await provider.status(deployment.deployment_id)
        print(f"   Status: {status.status}")
        print(f"   Uptime: {status.uptime_seconds:.1f} seconds")
        if status.metrics:
            print(f"   CPU: {status.metrics.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {status.metrics.get('memory_usage_mb', 0):.1f} MB")
        print()

        # Get logs
        print("📜 Recent logs (last 10 lines):")
        logs = await provider.logs(deployment.deployment_id, tail=10)
        for log in logs[-10:]:
            print(f"   {log}")
        print()

        # Keep running for 30 seconds
        print("⏳ Bot running for 30 seconds...")
        await asyncio.sleep(30)

    print("🛑 Deployment stopped (auto-cleanup on context exit)")
    print()


async def manual_deployment_example():
    """Manual deployment example without context manager."""
    print("=" * 60)
    print("Manual Deployment Example (No Auto-Cleanup)")
    print("=" * 60)
    print()

    config = DeploymentConfig(
        bot_name="Manual-Test-Bot",
        strategy_type="NFL",
        environment="sandbox",
    )

    provider = DockerDeploymentProvider()

    # Deploy manually
    print("🚀 Deploying...")
    result = await provider.deploy(config)
    print(f"✅ Deployed: {result.deployment_id}")
    print()

    try:
        # List all deployments
        print("📋 Active deployments:")
        deployments = await provider.list_deployments()
        for dep in deployments:
            print(f"   - {dep.bot_name} ({dep.status})")
        print()

        # Wait a bit
        await asyncio.sleep(5)

    finally:
        # Manual cleanup
        print("🛑 Stopping deployment...")
        await provider.stop(result.deployment_id)
        print("✅ Stopped successfully")
        print()


async def main():
    """Run all examples."""
    print("\n🤖 Neural SDK Deployment Module Examples\n")

    # Check if Docker is available
    try:
        provider = DockerDeploymentProvider()
        print("✅ Docker is available\n")
    except Exception as e:
        print(f"❌ Docker is not available: {e}")
        print("   Please make sure Docker is installed and running.")
        return

    # Run examples
    try:
        # Example 1: Basic deployment with context manager
        await basic_deployment_example()

        # Example 2: Manual deployment
        await manual_deployment_example()

        print("=" * 60)
        print("✨ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
