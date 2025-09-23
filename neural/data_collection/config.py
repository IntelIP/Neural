"""
Configuration management for the Neural SDK Data Collection Infrastructure.

This module provides a flexible configuration system that supports multiple
configuration sources (files, environment variables, defaults) and handles
validation, merging, and dynamic reloading of configurations.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import yaml
import logging
from dataclasses import asdict

from neural.data_collection.exceptions import ConfigurationError
from neural.data_collection.base import DataSourceConfig


# Configure module logger
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the Neural SDK data collection infrastructure.
    
    This class handles loading configurations from multiple sources,
    validating them against schemas, and providing a unified interface
    for accessing configuration values. It supports:
    
    - YAML and JSON configuration files
    - Environment variable substitution
    - Default values
    - Configuration validation
    - Hot-reloading (optional)
    
    Attributes:
        config_path: Path to the configuration file
        config_data: Loaded configuration data
        environment_prefix: Prefix for environment variables
        
    Example:
        >>> config_manager = ConfigManager("config.yaml")
        >>> websocket_config = config_manager.get_data_source_config("twitter_ws")
        >>> print(websocket_config.name)
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "neural": {
            "log_level": "INFO",
            "metrics_enabled": True,
            "max_concurrent_sources": 10
        },
        "data_sources": {}
    }
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        environment_prefix: str = "NEURAL_",
        auto_reload: bool = False
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            environment_prefix: Prefix for environment variables
            auto_reload: Whether to automatically reload on file changes
            
        Raises:
            ConfigurationError: If configuration cannot be loaded
            
        Example:
            >>> config = ConfigManager("config/neural.yaml")
        """
        self.config_path = Path(config_path) if config_path else None
        self.environment_prefix = environment_prefix
        self.auto_reload = auto_reload
        self.config_data: Dict[str, Any] = {}
        
        # Load configuration
        self.load_config()
        
        logger.info(
            f"ConfigManager initialized with config from: "
            f"{self.config_path or 'defaults and environment'}"
        )
    
    def load_config(self) -> None:
        """
        Load configuration from all sources and merge them.
        
        The configuration is loaded in the following priority order:
        1. Default configuration (lowest priority)
        2. Configuration file (if provided)
        3. Environment variables (highest priority)
        
        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        # Start with default configuration
        self.config_data = self._deep_copy(self.DEFAULT_CONFIG)
        
        # Load from file if provided
        if self.config_path and self.config_path.exists():
            file_config = self._load_from_file(self.config_path)
            self.config_data = self._merge_configs(self.config_data, file_config)
            logger.debug(f"Loaded configuration from {self.config_path}")
        elif self.config_path:
            logger.warning(f"Configuration file not found: {self.config_path}")
        
        # Apply environment variables
        self._apply_environment_variables()
        
        # Validate the final configuration
        self._validate_config()
        
        logger.debug("Configuration loaded and validated successfully")
    
    def _load_from_file(self, path: Path) -> Dict[str, Any]:
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix == '.json':
                    return json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {path.suffix}",
                        details={"file": str(path)}
                    )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {e}",
                details={"file": str(path)}
            )
    
    def _apply_environment_variables(self) -> None:
        """
        Apply environment variables to the configuration.
        
        Environment variables are processed in two ways:
        1. Direct override: NEURAL_<SECTION>_<KEY>=value
        2. Template substitution: ${ENV_VAR} in config values
        """
        # Process template substitutions in existing config
        self._substitute_env_vars(self.config_data)
        
        # Apply direct environment variable overrides
        for env_key, env_value in os.environ.items():
            if env_key.startswith(self.environment_prefix):
                # Remove prefix and convert to lowercase
                config_key = env_key[len(self.environment_prefix):].lower()
                
                # Split by underscore to create nested keys
                keys = config_key.split('_')
                
                # Navigate to the correct position in config
                current = self.config_data
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value, attempting type conversion
                current[keys[-1]] = self._parse_env_value(env_value)
                logger.debug(f"Applied environment variable: {env_key}")
    
    def _substitute_env_vars(self, config: Union[Dict, List, str]) -> None:
        """
        Recursively substitute environment variables in configuration values.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            config: Configuration data to process
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str):
                    config[key] = self._expand_env_var(value)
                else:
                    self._substitute_env_vars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, str):
                    config[i] = self._expand_env_var(item)
                else:
                    self._substitute_env_vars(item)
    
    def _expand_env_var(self, value: str) -> str:
        """
        Expand environment variables in a string value.
        
        Args:
            value: String potentially containing ${VAR} placeholders
            
        Returns:
            String with environment variables expanded
        """
        # Pattern to match ${VAR} or ${VAR:default}
        pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) or ""
            return os.environ.get(var_name, default_value)
        
        return pattern.sub(replacer, value)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment
            
        Returns:
            Parsed value with appropriate type
        """
        # Try to parse as boolean
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Configuration to merge on top
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _deep_copy(self, obj: Any) -> Any:
        """
        Create a deep copy of a configuration object.
        
        Args:
            obj: Object to copy
            
        Returns:
            Deep copy of the object
        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Check for required top-level keys
        required_keys = ["neural"]
        for key in required_keys:
            if key not in self.config_data:
                raise ConfigurationError(
                    f"Missing required configuration section: {key}"
                )
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = self.config_data.get("neural", {}).get("log_level", "INFO")
        if log_level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log level: {log_level}",
                details={"valid_levels": valid_log_levels}
            )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> value = config.get("neural.log_level", "INFO")
        """
        keys = key_path.split('.')
        current = self.config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
            
        Example:
            >>> config.set("neural.log_level", "DEBUG")
        """
        keys = key_path.split('.')
        current = self.config_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def get_data_source_config(
        self,
        source_name: str,
        config_class: Type[DataSourceConfig] = DataSourceConfig
    ) -> DataSourceConfig:
        """
        Get configuration for a specific data source.
        
        Args:
            source_name: Name of the data source
            config_class: Class to instantiate with configuration
            
        Returns:
            Data source configuration object
            
        Raises:
            ConfigurationError: If data source not found
            
        Example:
            >>> ws_config = config.get_data_source_config("twitter_ws")
        """
        sources = self.config_data.get("data_sources", {})
        
        if source_name not in sources:
            raise ConfigurationError(
                f"Data source not found: {source_name}",
                details={"available_sources": list(sources.keys())}
            )
        
        source_config = sources[source_name]
        
        # Add the source name if not present
        if "name" not in source_config:
            source_config["name"] = source_name
        
        try:
            # Create configuration object
            return config_class(**source_config)
        except TypeError as e:
            raise ConfigurationError(
                f"Invalid configuration for data source: {source_name}",
                details={"error": str(e)}
            )
    
    def list_data_sources(self) -> List[str]:
        """
        List all configured data sources.
        
        Returns:
            List of data source names
            
        Example:
            >>> sources = config.list_data_sources()
            >>> print(f"Available sources: {sources}")
        """
        return list(self.config_data.get("data_sources", {}).keys())
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to a file.
        
        Args:
            path: Path to save configuration (uses original path if not provided)
            
        Raises:
            ConfigurationError: If configuration cannot be saved
            
        Example:
            >>> config.save("config/backup.yaml")
        """
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            raise ConfigurationError("No path provided for saving configuration")
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    yaml.safe_dump(self.config_data, f, default_flow_style=False)
                elif save_path.suffix == '.json':
                    json.dump(self.config_data, f, indent=2)
                else:
                    raise ConfigurationError(
                        f"Unsupported file format: {save_path.suffix}"
                    )
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}",
                details={"path": str(save_path)}
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Complete configuration dictionary
            
        Example:
            >>> config_dict = config.to_dict()
        """
        return self._deep_copy(self.config_data)
    
    def __repr__(self) -> str:
        """Return string representation of the configuration manager."""
        return (
            f"ConfigManager("
            f"config_path={self.config_path}, "
            f"sources={len(self.list_data_sources())})"
        )