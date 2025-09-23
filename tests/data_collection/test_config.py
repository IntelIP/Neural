"""
Tests for configuration management.
"""

import os
import json
import yaml
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from neural.data_collection.config import ConfigManager
from neural.data_collection.exceptions import ConfigurationError


class TestConfigManager:
    """Test ConfigManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()
    
    def test_default_initialization(self, config_manager):
        assert config_manager.config == {}
        assert config_manager._config_file is None
        assert config_manager._env_prefix == "NEURAL_"
    
    def test_custom_initialization(self):
        manager = ConfigManager(
            config_file="config.yaml",
            env_prefix="TEST_",
            defaults={"key": "value"}
        )
        
        assert manager._config_file == "config.yaml"
        assert manager._env_prefix == "TEST_"
        assert manager.config == {"key": "value"}
    
    def test_load_yaml_config(self, temp_dir):
        # Create YAML config file
        config_path = Path(temp_dir) / "config.yaml"
        config_data = {
            "api": {
                "url": "https://api.example.com",
                "timeout": 30
            },
            "features": {
                "cache": True,
                "retry": True
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(config_file=str(config_path))
        manager.load_config()
        
        assert manager.config["api"]["url"] == "https://api.example.com"
        assert manager.config["api"]["timeout"] == 30
        assert manager.config["features"]["cache"] is True
    
    def test_load_json_config(self, temp_dir):
        # Create JSON config file
        config_path = Path(temp_dir) / "config.json"
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigManager(config_file=str(config_path))
        manager.load_config()
        
        assert manager.config["database"]["host"] == "localhost"
        assert manager.config["database"]["port"] == 5432
    
    def test_environment_variable_substitution(self, temp_dir):
        # Set environment variables
        os.environ["TEST_API_KEY"] = "secret123"
        os.environ["TEST_PORT"] = "8080"
        
        # Create config with env var references
        config_path = Path(temp_dir) / "config.yaml"
        config_data = {
            "api_key": "${TEST_API_KEY}",
            "port": "${TEST_PORT}",
            "default_value": "${MISSING_VAR:default}"
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        try:
            manager = ConfigManager(config_file=str(config_path))
            manager.load_config()
            
            assert manager.config["api_key"] == "secret123"
            assert manager.config["port"] == "8080"
            assert manager.config["default_value"] == "default"
        finally:
            # Clean up
            del os.environ["TEST_API_KEY"]
            del os.environ["TEST_PORT"]
    
    def test_environment_override(self):
        # Test that environment variables override config file
        os.environ["NEURAL_API_URL"] = "https://override.com"
        
        try:
            manager = ConfigManager(
                defaults={"api": {"url": "https://default.com"}}
            )
            manager.load_config()
            
            assert manager.get("api.url") == "https://override.com"
        finally:
            del os.environ["NEURAL_API_URL"]
    
    def test_get_nested_value(self, config_manager):
        config_manager.config = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        assert config_manager.get("level1.level2.level3") == "value"
        assert config_manager.get("level1.level2") == {"level3": "value"}
    
    def test_get_with_default(self, config_manager):
        config_manager.config = {"key": "value"}
        
        assert config_manager.get("key") == "value"
        assert config_manager.get("missing", default="default") == "default"
        assert config_manager.get("missing") is None
    
    def test_set_value(self, config_manager):
        config_manager.set("api.key", "secret")
        
        assert config_manager.config["api"]["key"] == "secret"
    
    def test_set_nested_value(self, config_manager):
        config_manager.set("a.b.c.d", "deep_value")
        
        assert config_manager.config["a"]["b"]["c"]["d"] == "deep_value"
    
    def test_update_config(self, config_manager):
        config_manager.config = {"existing": "value"}
        
        config_manager.update({
            "new": "data",
            "nested": {"key": "value"}
        })
        
        assert config_manager.config["existing"] == "value"
        assert config_manager.config["new"] == "data"
        assert config_manager.config["nested"]["key"] == "value"
    
    def test_validate_required_fields(self, config_manager):
        config_manager.config = {
            "api": {"key": "secret"},
            "database": {"host": "localhost"}
        }
        
        # Should not raise with all required fields present
        config_manager.validate(required=["api.key", "database.host"])
        
        # Should raise with missing field
        with pytest.raises(ConfigurationError) as exc:
            config_manager.validate(required=["api.key", "missing.field"])
        
        assert "missing.field" in str(exc.value)
    
    def test_save_config_yaml(self, temp_dir, config_manager):
        save_path = Path(temp_dir) / "saved.yaml"
        
        config_manager.config = {
            "test": {"data": "value"},
            "number": 42
        }
        
        config_manager.save(str(save_path))
        
        # Load and verify
        with open(save_path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded["test"]["data"] == "value"
        assert loaded["number"] == 42
    
    def test_save_config_json(self, temp_dir, config_manager):
        save_path = Path(temp_dir) / "saved.json"
        
        config_manager.config = {
            "test": {"data": "value"},
            "number": 42
        }
        
        config_manager.save(str(save_path))
        
        # Load and verify
        with open(save_path) as f:
            loaded = json.load(f)
        
        assert loaded["test"]["data"] == "value"
        assert loaded["number"] == 42
    
    def test_reload_config(self, temp_dir):
        config_path = Path(temp_dir) / "config.yaml"
        
        # Initial config
        with open(config_path, 'w') as f:
            yaml.dump({"version": 1}, f)
        
        manager = ConfigManager(config_file=str(config_path))
        manager.load_config()
        
        assert manager.config["version"] == 1
        
        # Update config file
        with open(config_path, 'w') as f:
            yaml.dump({"version": 2, "new": "field"}, f)
        
        manager.reload()
        
        assert manager.config["version"] == 2
        assert manager.config["new"] == "field"
    
    def test_clear_config(self, config_manager):
        config_manager.config = {"data": "value"}
        config_manager.clear()
        
        assert config_manager.config == {}
    
    def test_nonexistent_config_file(self):
        manager = ConfigManager(config_file="nonexistent.yaml")
        
        # Should not raise, just log warning
        manager.load_config()
        
        assert manager.config == {}
    
    def test_invalid_yaml_file(self, temp_dir):
        config_path = Path(temp_dir) / "invalid.yaml"
        
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content:")
        
        manager = ConfigManager(config_file=str(config_path))
        
        with pytest.raises(ConfigurationError):
            manager.load_config()
    
    def test_merge_defaults(self):
        defaults = {
            "api": {"timeout": 30},
            "features": {"cache": True}
        }
        
        manager = ConfigManager(defaults=defaults)
        manager.update({"api": {"url": "https://api.com"}})
        
        # Should merge, not replace
        assert manager.config["api"]["timeout"] == 30
        assert manager.config["api"]["url"] == "https://api.com"
        assert manager.config["features"]["cache"] is True