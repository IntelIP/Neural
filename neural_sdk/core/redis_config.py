"""
Environment-specific Redis Configuration

Manages Redis connections and configurations for different environments
with proper isolation and safety mechanisms.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.asyncio.sentinel import Sentinel

from .environment_manager import Environment, EnvironmentManager


@dataclass
class RedisConfig:
    """Redis configuration for an environment."""

    host: str
    port: int
    db: int
    password: Optional[str]
    prefix: str
    max_connections: int
    socket_timeout: int
    socket_connect_timeout: int
    socket_keepalive: bool
    socket_keepalive_options: Dict[str, Any]
    connection_pool_class: str
    ssl: bool
    ssl_certfile: Optional[str]
    ssl_keyfile: Optional[str]
    ssl_ca_certs: Optional[str]
    sentinel_hosts: List[Tuple[str, int]]
    sentinel_service_name: Optional[str]
    cluster_mode: bool

    @classmethod
    def for_environment(cls, env: Environment) -> "RedisConfig":
        """Get Redis configuration for environment."""
        base_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "password": os.getenv("REDIS_PASSWORD"),
            "max_connections": 50,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
            "connection_pool_class": "BlockingConnectionPool",
            "ssl": False,
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "ssl_ca_certs": None,
            "sentinel_hosts": [],
            "sentinel_service_name": None,
            "cluster_mode": False,
        }

        # Environment-specific configurations
        if env == Environment.PRODUCTION:
            config = base_config.copy()
            config.update(
                {
                    "db": 0,
                    "prefix": "prod:",
                    "max_connections": 100,
                    "ssl": True,
                    "ssl_ca_certs": os.getenv("REDIS_SSL_CA"),
                    "sentinel_hosts": cls._parse_sentinel_hosts(
                        os.getenv("REDIS_SENTINELS", "")
                    ),
                    "sentinel_service_name": "production-master",
                    "cluster_mode": True,
                }
            )
            return cls(**config)

        elif env == Environment.STAGING:
            config = base_config.copy()
            config.update(
                {
                    "db": 1,
                    "prefix": "staging:",
                    "max_connections": 75,
                    "ssl": True,
                    "ssl_ca_certs": os.getenv("REDIS_SSL_CA"),
                }
            )
            return cls(**config)

        elif env == Environment.SANDBOX:
            config = base_config.copy()
            config.update({"db": 2, "prefix": "sandbox:", "max_connections": 50})
            return cls(**config)

        elif env == Environment.TRAINING:
            config = base_config.copy()
            config.update(
                {
                    "db": 3,
                    "prefix": "training:",
                    "max_connections": 200,  # Higher for parallel training
                    "socket_timeout": 10,  # Longer timeout for training ops
                }
            )
            return cls(**config)

        else:  # DEVELOPMENT
            config = base_config.copy()
            config.update({"db": 4, "prefix": "dev:", "max_connections": 25})
            return cls(**config)

    @staticmethod
    def _parse_sentinel_hosts(hosts_str: str) -> List[Tuple[str, int]]:
        """Parse sentinel hosts from string."""
        hosts = []
        if hosts_str:
            for host in hosts_str.split(","):
                if ":" in host:
                    h, p = host.split(":")
                    hosts.append((h.strip(), int(p)))
        return hosts

    def to_connection_kwargs(self) -> Dict[str, Any]:
        """Convert to Redis connection kwargs."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_keepalive": self.socket_keepalive,
            "socket_keepalive_options": self.socket_keepalive_options,
            "max_connections": self.max_connections,
        }

        if self.ssl:
            kwargs["ssl"] = True
            if self.ssl_certfile:
                kwargs["ssl_certfile"] = self.ssl_certfile
            if self.ssl_keyfile:
                kwargs["ssl_keyfile"] = self.ssl_keyfile
            if self.ssl_ca_certs:
                kwargs["ssl_ca_certs"] = self.ssl_ca_certs

        return kwargs


class EnvironmentRedisManager:
    """Manages Redis connections for different environments."""

    def __init__(self, env_manager: EnvironmentManager):
        """Initialize Redis manager."""
        self.env_manager = env_manager
        self._connections: Dict[Environment, redis.Redis] = {}
        self._pools: Dict[Environment, ConnectionPool] = {}
        self._sentinels: Dict[Environment, Sentinel] = {}
        self._pubsub_connections: Dict[str, redis.client.PubSub] = {}
        self._channel_isolation: bool = True
        self._allowed_channels: Dict[Environment, Set[str]] = {
            Environment.PRODUCTION: {"prod:*", "market:*", "trade:*", "risk:*"},
            Environment.STAGING: {"staging:*", "test:*", "market:*"},
            Environment.SANDBOX: {"sandbox:*", "demo:*", "test:*"},
            Environment.TRAINING: {"training:*", "synthetic:*", "replay:*"},
            Environment.DEVELOPMENT: {"*"},  # All channels allowed in dev
        }

    async def get_connection(
        self, environment: Optional[Environment] = None
    ) -> redis.Redis:
        """Get Redis connection for environment."""
        env = environment or self.env_manager.get_current_environment()

        if env not in self._connections:
            config = RedisConfig.for_environment(env)

            # Use sentinel if configured
            if config.sentinel_hosts:
                conn = await self._create_sentinel_connection(env, config)
            # Use cluster if configured
            elif config.cluster_mode:
                conn = await self._create_cluster_connection(env, config)
            # Use standard connection
            else:
                conn = await self._create_standard_connection(env, config)

            self._connections[env] = conn

        return self._connections[env]

    async def _create_standard_connection(
        self, env: Environment, config: RedisConfig
    ) -> redis.Redis:
        """Create standard Redis connection."""
        # Create connection pool
        if env not in self._pools:
            pool_class = getattr(redis, config.connection_pool_class)
            self._pools[env] = pool_class(**config.to_connection_kwargs())

        return redis.Redis(connection_pool=self._pools[env])

    async def _create_sentinel_connection(
        self, env: Environment, config: RedisConfig
    ) -> redis.Redis:
        """Create sentinel-based Redis connection."""
        if env not in self._sentinels:
            self._sentinels[env] = Sentinel(
                config.sentinel_hosts, socket_timeout=config.socket_timeout
            )

        # Get master from sentinel
        master = self._sentinels[env].master_for(
            config.sentinel_service_name,
            socket_timeout=config.socket_timeout,
            db=config.db,
            password=config.password,
        )

        return master

    async def _create_cluster_connection(
        self, env: Environment, config: RedisConfig
    ) -> redis.Redis:
        """Create cluster Redis connection."""
        # Simplified - in production use redis.cluster.RedisCluster
        return await self._create_standard_connection(env, config)

    async def get_pubsub(
        self, channels: List[str], environment: Optional[Environment] = None
    ) -> redis.client.PubSub:
        """Get pub/sub connection for channels."""
        env = environment or self.env_manager.get_current_environment()

        # Validate channels if isolation enabled
        if self._channel_isolation:
            valid, invalid = self._validate_channels(env, channels)
            if invalid:
                raise ValueError(f"Channels not allowed in {env.value}: {invalid}")

        # Create unique key for this subscription
        sub_key = f"{env.value}:{','.join(sorted(channels))}"

        if sub_key not in self._pubsub_connections:
            conn = await self.get_connection(env)
            pubsub = conn.pubsub()

            # Subscribe to channels
            for channel in channels:
                await pubsub.subscribe(channel)

            self._pubsub_connections[sub_key] = pubsub

        return self._pubsub_connections[sub_key]

    def _validate_channels(
        self, env: Environment, channels: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Validate channels for environment."""
        allowed = self._allowed_channels.get(env, set())
        valid = []
        invalid = []

        for channel in channels:
            # Check if channel matches any allowed pattern
            is_valid = False
            for pattern in allowed:
                if pattern == "*" or self._matches_pattern(channel, pattern):
                    is_valid = True
                    break

            if is_valid:
                valid.append(channel)
            else:
                invalid.append(channel)

        return valid, invalid

    def _matches_pattern(self, channel: str, pattern: str) -> bool:
        """Check if channel matches pattern."""
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            return channel.startswith(prefix)
        return channel == pattern

    async def publish(
        self, channel: str, message: Any, environment: Optional[Environment] = None
    ) -> int:
        """Publish message to channel."""
        env = environment or self.env_manager.get_current_environment()

        # Validate channel
        if self._channel_isolation:
            valid, _ = self._validate_channels(env, [channel])
            if not valid:
                raise ValueError(f"Channel '{channel}' not allowed in {env.value}")

        # Get connection and publish
        conn = await self.get_connection(env)

        # Add environment prefix if not present
        config = RedisConfig.for_environment(env)
        if not channel.startswith(config.prefix):
            channel = f"{config.prefix}{channel}"

        # Serialize message
        if not isinstance(message, (str, bytes)):
            message = json.dumps(message)

        return await conn.publish(channel, message)

    async def get_key(
        self, key: str, environment: Optional[Environment] = None
    ) -> Optional[str]:
        """Get value for key."""
        env = environment or self.env_manager.get_current_environment()
        conn = await self.get_connection(env)

        # Add environment prefix
        config = RedisConfig.for_environment(env)
        if not key.startswith(config.prefix):
            key = f"{config.prefix}{key}"

        return await conn.get(key)

    async def set_key(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        environment: Optional[Environment] = None,
    ) -> bool:
        """Set value for key."""
        env = environment or self.env_manager.get_current_environment()
        conn = await self.get_connection(env)

        # Add environment prefix
        config = RedisConfig.for_environment(env)
        if not key.startswith(config.prefix):
            key = f"{config.prefix}{key}"

        # Serialize value if needed
        if not isinstance(value, (str, bytes)):
            value = json.dumps(value)

        if ttl:
            return await conn.setex(key, ttl, value)
        else:
            return await conn.set(key, value)

    async def delete_key(
        self, key: str, environment: Optional[Environment] = None
    ) -> int:
        """Delete key."""
        env = environment or self.env_manager.get_current_environment()

        # Check if deletion is allowed
        if env == Environment.PRODUCTION:
            raise PermissionError("Key deletion not allowed in production")

        conn = await self.get_connection(env)

        # Add environment prefix
        config = RedisConfig.for_environment(env)
        if not key.startswith(config.prefix):
            key = f"{config.prefix}{key}"

        return await conn.delete(key)

    async def flush_db(
        self, environment: Optional[Environment] = None, force: bool = False
    ) -> bool:
        """Flush database for environment."""
        env = environment or self.env_manager.get_current_environment()

        # Safety checks
        if env == Environment.PRODUCTION and not force:
            raise PermissionError("Cannot flush production database without force flag")

        if env in [Environment.PRODUCTION, Environment.STAGING]:
            # Require additional confirmation
            confirm = input(f"Type 'FLUSH {env.value.upper()}' to confirm: ")
            if confirm != f"FLUSH {env.value.upper()}":
                return False

        conn = await self.get_connection(env)
        await conn.flushdb()
        return True

    async def get_info(
        self, section: Optional[str] = None, environment: Optional[Environment] = None
    ) -> Dict[str, Any]:
        """Get Redis server info."""
        env = environment or self.env_manager.get_current_environment()
        conn = await self.get_connection(env)

        info = await conn.info(section) if section else await conn.info()
        return info

    async def ping(self, environment: Optional[Environment] = None) -> bool:
        """Ping Redis server."""
        env = environment or self.env_manager.get_current_environment()
        conn = await self.get_connection(env)

        try:
            return await conn.ping()
        except Exception:
            return False

    async def close_all(self) -> None:
        """Close all connections."""
        # Close pub/sub connections
        for pubsub in self._pubsub_connections.values():
            await pubsub.close()
        self._pubsub_connections.clear()

        # Close regular connections
        for conn in self._connections.values():
            await conn.close()
        self._connections.clear()

        # Close pools
        for pool in self._pools.values():
            await pool.disconnect()
        self._pools.clear()

    async def migrate_data(
        self,
        from_env: Environment,
        to_env: Environment,
        pattern: str = "*",
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Migrate data between environments."""
        # Safety check
        if to_env == Environment.PRODUCTION and not dry_run:
            raise PermissionError("Cannot migrate to production without dry run first")

        from_conn = await self.get_connection(from_env)
        to_conn = await self.get_connection(to_env)

        from_config = RedisConfig.for_environment(from_env)
        to_config = RedisConfig.for_environment(to_env)

        # Find keys to migrate
        search_pattern = f"{from_config.prefix}{pattern}"
        keys = []
        cursor = 0

        while True:
            cursor, batch = await from_conn.scan(
                cursor, match=search_pattern, count=100
            )
            keys.extend(batch)
            if cursor == 0:
                break

        migrated = []
        errors = []

        for key in keys:
            try:
                # Get value and TTL
                value = await from_conn.get(key)
                ttl = await from_conn.ttl(key)

                if not dry_run:
                    # Transform key for new environment
                    new_key = key.replace(from_config.prefix, to_config.prefix)

                    # Set in new environment
                    if ttl > 0:
                        await to_conn.setex(new_key, ttl, value)
                    else:
                        await to_conn.set(new_key, value)

                migrated.append(key)

            except Exception as e:
                errors.append({"key": key, "error": str(e)})

        return {
            "total_keys": len(keys),
            "migrated": len(migrated),
            "errors": len(errors),
            "dry_run": dry_run,
            "from_env": from_env.value,
            "to_env": to_env.value,
            "error_details": errors[:10],  # First 10 errors
        }

    def get_channel_prefix(self, environment: Optional[Environment] = None) -> str:
        """Get channel prefix for environment."""
        env = environment or self.env_manager.get_current_environment()
        config = RedisConfig.for_environment(env)
        return config.prefix

    def is_channel_allowed(
        self, channel: str, environment: Optional[Environment] = None
    ) -> bool:
        """Check if channel is allowed in environment."""
        env = environment or self.env_manager.get_current_environment()
        valid, _ = self._validate_channels(env, [channel])
        return len(valid) > 0
