"""
Kalshi WebSocket Infrastructure - Configuration Settings
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables - try multiple .env files in order of preference
project_root = Path(__file__).parent.parent.parent.parent
env_files = ['.env', '.env.demo', '.env.production']

for env_file in env_files:
    env_path = project_root / env_file
    if env_path.exists():
        load_dotenv(env_path)
        break


@dataclass
class KalshiConfig:
    """Configuration for Kalshi WebSocket infrastructure"""
    
    # API Credentials
    api_key_id: str
    private_key: str
    
    # Environment Settings
    environment: str  # 'demo' or 'prod'
    
    # API URLs
    api_base_url: str
    ws_url: str
    
    # Connection Settings
    heartbeat_interval: int = 30  # seconds
    reconnect_attempts: int = 5
    reconnect_delay: int = 5  # seconds
    
    # Subscription Settings
    max_subscriptions: int = 100
    batch_size: int = 10  # for batch operations
    
    # Data Settings
    price_precision: int = 4  # decimal places for price conversion
    
    @property
    def is_demo(self) -> bool:
        """Check if running in demo environment"""
        return self.environment == 'demo'
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == 'prod'


def get_config() -> KalshiConfig:
    """
    Get Kalshi configuration from environment variables
    
    Returns:
        KalshiConfig: Configuration object with all settings
    """
    # Get environment
    environment = os.getenv('KALSHI_ENVIRONMENT', 'demo').lower()
    
    # Set URLs based on environment
    if environment == 'prod':
        api_base_url = 'https://api.elections.kalshi.com/trade-api/v2/'
        ws_url = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
    else:  # demo
        api_base_url = 'https://demo-api.kalshi.co/trade-api/v2/'
        ws_url = 'wss://demo-api.kalshi.co/trade-api/ws/v2'
    
    # Override with explicit URL if provided
    api_base_url = os.getenv('KALSHI_API_BASE', api_base_url)
    ws_url = os.getenv('KALSHI_WS_URL', ws_url)
    
    # Get credentials
    api_key_id = os.getenv('KALSHI_API_KEY_ID')
    private_key = os.getenv('KALSHI_PRIVATE_KEY')
    
    # Support file-based private key via KALSHI_PRIVATE_KEY_FILE
    if not private_key:
        key_file = os.getenv('KALSHI_PRIVATE_KEY_FILE')
        if key_file:
            # Handle both absolute and relative paths
            key_path = Path(key_file)
            if not key_path.is_absolute():
                # Make relative paths relative to project root
                key_path = Path(__file__).parent.parent.parent.parent / key_path
            
            # Expand user home directory if present
            key_path = key_path.expanduser()
            
            if key_path.exists():
                try:
                    private_key = key_path.read_text().strip()
                except Exception as e:
                    raise ValueError(f"Failed to read private key from {key_path}: {e}")
            else:
                # Provide helpful error message
                available_keys = list(Path(__file__).parent.parent.parent.parent.glob("keys/*.key"))
                if available_keys:
                    key_names = [k.name for k in available_keys]
                    raise ValueError(
                        f"Private key file not found: {key_path}\n"
                        f"Available keys in keys/ directory: {', '.join(key_names)}\n"
                        f"Please check KALSHI_PRIVATE_KEY_FILE in .env"
                    )
                else:
                    raise ValueError(
                        f"Private key file not found: {key_path}\n"
                        "No keys found in keys/ directory. Please add your private key files."
                    )
    
    if not api_key_id or not private_key:
        raise ValueError(
            "Missing required Kalshi credentials. "
            "Please set KALSHI_API_KEY_ID and either KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_FILE environment variables."
        )
    
    return KalshiConfig(
        api_key_id=api_key_id,
        private_key=private_key,
        environment=environment,
        api_base_url=api_base_url,
        ws_url=ws_url,
        heartbeat_interval=int(os.getenv('KALSHI_HEARTBEAT_INTERVAL', '30')),
        reconnect_attempts=int(os.getenv('KALSHI_RECONNECT_ATTEMPTS', '5')),
        reconnect_delay=int(os.getenv('KALSHI_RECONNECT_DELAY', '5')),
        max_subscriptions=int(os.getenv('KALSHI_MAX_SUBSCRIPTIONS', '100')),
        batch_size=int(os.getenv('KALSHI_BATCH_SIZE', '10')),
        price_precision=int(os.getenv('KALSHI_PRICE_PRECISION', '4'))
    )


# Create a singleton config instance
_config: Optional[KalshiConfig] = None


def get_cached_config() -> KalshiConfig:
    """Get cached configuration instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = get_config()
    return _config
