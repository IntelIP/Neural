"""
Kalshi WebSocket Infrastructure - Authentication Module
Implements RSA-PSS authentication for Kalshi API
"""

import base64
import time
from typing import Dict, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from ...config.settings import KalshiConfig


class KalshiAuth:
    """Handle RSA-PSS authentication for Kalshi API"""
    
    def __init__(self, config: Optional[KalshiConfig] = None):
        """
        Initialize authentication with configuration
        
        Args:
            config: Optional KalshiConfig instance. If not provided, will get from environment
        """
        if config is None:
            # Use environment to create config
            import os
            from pathlib import Path
            from dotenv import load_dotenv
            
            env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
            load_dotenv(env_path)
            
            from ...config.settings import KalshiConfig
            
            environment = os.getenv("KALSHI_ENVIRONMENT", "prod")
            api_base_url = "https://api.elections.kalshi.com/trade-api/v2" if environment == "prod" else "https://demo-api.kalshi.co/trade-api/v2"
            
            # Load private key from file  
            private_key_file = os.getenv("KALSHI_PRIVATE_KEY_FILE", "./keys/kalshi_prod_private.key")
            if not private_key_file.startswith('/'):
                # Relative path - resolve from project root
                key_path = Path(__file__).parent.parent.parent.parent.parent / private_key_file
            else:
                # Absolute path
                key_path = Path(private_key_file)
            
            with open(key_path, 'r') as f:
                private_key = f.read()
            
            config = KalshiConfig(
                api_key_id=os.getenv("KALSHI_API_KEY_ID"),
                private_key=private_key,
                environment=environment,
                api_base_url=api_base_url
            )
        
        self.config = config
        self.api_key_id = config.api_key_id
        self._load_private_key(config.private_key)
    
    def _load_private_key(self, private_key_str: str) -> None:
        """
        Load and parse the RSA private key
        
        Args:
            private_key_str: PEM formatted private key string
        """
        # Normalize common formatting issues
        private_key_str = (private_key_str or "").strip().strip('"').strip("'")
        # Convert escaped newlines ("\n") to real newlines if needed
        if "\\n" in private_key_str and "\n" not in private_key_str:
            private_key_str = private_key_str.replace("\\n", "\n")
        # Add PEM headers if not present
        if not private_key_str.startswith('-----BEGIN'):
            private_key_str = f"-----BEGIN RSA PRIVATE KEY-----\n{private_key_str}\n-----END RSA PRIVATE KEY-----"
        
        self.private_key = serialization.load_pem_private_key(
            private_key_str.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
    
    def sign_request(self, method: str, path: str, timestamp: Optional[str] = None) -> str:
        """
        Sign a request using RSA-PSS with SHA256
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., '/trade-api/v2/markets')
            timestamp: Optional timestamp string. If not provided, current time is used
        
        Returns:
            Base64 encoded signature
        """
        if timestamp is None:
            timestamp = str(int(time.time()))
        
        # Create message to sign
        message = f"{timestamp}{method}{path}".encode('utf-8')
        
        # Sign with RSA-PSS
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('utf-8')
    
    def get_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Get authentication headers for a request
        
        Args:
            method: HTTP method
            path: API path
        
        Returns:
            Dictionary of authentication headers
        """
        timestamp = str(int(time.time()))
        signature = self.sign_request(method, path, timestamp)
        
        return {
            'KALSHI-ACCESS-KEY': self.api_key_id,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': timestamp
        }
    
    def get_websocket_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for WebSocket connection
        
        Returns:
            Dictionary of WebSocket authentication headers
        """
        # WebSocket connections use GET method with root path
        return self.get_auth_headers('GET', '/')
    
    def validate_timestamp(self, timestamp: str, max_age_seconds: int = 30) -> bool:
        """
        Validate that a timestamp is recent enough
        
        Args:
            timestamp: Timestamp string to validate
            max_age_seconds: Maximum age in seconds
        
        Returns:
            True if timestamp is valid and recent
        """
        try:
            ts = int(timestamp)
            current_time = int(time.time())
            age = abs(current_time - ts)
            return age <= max_age_seconds
        except (ValueError, TypeError):
            return False
