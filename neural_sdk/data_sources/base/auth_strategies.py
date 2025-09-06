"""
Authentication Strategies for REST Data Sources

Provides various authentication methods for different APIs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import base64
import hashlib
import hmac
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class AuthStrategy(ABC):
    """Abstract base class for authentication strategies."""
    
    @abstractmethod
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """
        Get authentication headers for request.
        
        Args:
            method: HTTP method
            path: Request path
            
        Returns:
            Dictionary of headers
        """
        pass


class NoAuth(AuthStrategy):
    """No authentication required."""
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Return empty headers for no auth."""
        return {}


class APIKeyAuth(AuthStrategy):
    """API key authentication in header or query parameter."""
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key", in_header: bool = True):
        """
        Initialize API key authentication.
        
        Args:
            api_key: The API key
            header_name: Name of the header field
            in_header: If True, put in header; if False, put in query params
        """
        self.api_key = api_key
        self.header_name = header_name
        self.in_header = in_header
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Get API key headers."""
        if self.in_header:
            return {self.header_name: self.api_key}
        return {}
    
    def get_params(self) -> Dict[str, str]:
        """Get API key as query parameter."""
        if not self.in_header:
            return {"api_key": self.api_key}
        return {}


class BearerTokenAuth(AuthStrategy):
    """Bearer token authentication (OAuth 2.0 style)."""
    
    def __init__(self, token: str, prefix: str = "Bearer"):
        """
        Initialize bearer token authentication.
        
        Args:
            token: The bearer token
            prefix: Token prefix (usually "Bearer")
        """
        self.token = token
        self.prefix = prefix
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Get bearer token headers."""
        return {"Authorization": f"{self.prefix} {self.token}"}


class BasicAuth(AuthStrategy):
    """HTTP Basic authentication."""
    
    def __init__(self, username: str, password: str):
        """
        Initialize basic authentication.
        
        Args:
            username: Username
            password: Password
        """
        self.username = username
        self.password = password
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Get basic auth headers."""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}


class HMACAuth(AuthStrategy):
    """HMAC-based authentication."""
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Initialize HMAC authentication.
        
        Args:
            api_key: API key
            secret_key: Secret key for HMAC
        """
        self.api_key = api_key
        self.secret_key = secret_key
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Get HMAC auth headers."""
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{method}{path}{timestamp}"
        
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Signature": signature
        }


class RSASignatureAuth(AuthStrategy):
    """
    RSA-PSS signature authentication (Kalshi-style).
    
    This is specifically designed for APIs that require
    RSA-PSS signatures like Kalshi's API.
    """
    
    def __init__(self, api_key_id: str, private_key_str: str):
        """
        Initialize RSA signature authentication.
        
        Args:
            api_key_id: API key identifier
            private_key_str: PEM-encoded private key string
        """
        self.api_key_id = api_key_id
        
        # Load private key
        self.private_key = serialization.load_pem_private_key(
            private_key_str.encode(),
            password=None
        )
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """
        Get RSA signature headers.
        
        This follows Kalshi's signature scheme:
        1. Create message from timestamp + method + path
        2. Sign with RSA-PSS
        3. Include in headers
        """
        timestamp_ms = str(int(datetime.utcnow().timestamp() * 1000))
        message = f"{timestamp_ms}{method}{path}"
        
        # Sign message with RSA-PSS
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Encode signature as base64
        signature_b64 = base64.b64encode(signature).decode()
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms
        }


class OAuth2Auth(AuthStrategy):
    """
    OAuth 2.0 authentication with automatic token refresh.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None
    ):
        """
        Initialize OAuth 2.0 authentication.
        
        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            token_url: Token endpoint URL
            scope: Optional scope string
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.access_token = None
        self.token_expiry = None
    
    async def refresh_token(self):
        """Refresh the access token."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
            
            if self.scope:
                data["scope"] = self.scope
            
            response = await client.post(self.token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            
            # Calculate expiry time
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Get OAuth 2.0 headers with automatic refresh."""
        # Refresh token if needed
        if not self.access_token or datetime.utcnow() >= self.token_expiry:
            await self.refresh_token()
        
        return {"Authorization": f"Bearer {self.access_token}"}


class CustomHeaderAuth(AuthStrategy):
    """Custom header-based authentication."""
    
    def __init__(self, headers: Dict[str, str]):
        """
        Initialize custom header authentication.
        
        Args:
            headers: Dictionary of custom headers
        """
        self.headers = headers
    
    async def get_headers(self, method: str = "GET", path: str = "/") -> Dict[str, str]:
        """Return custom headers."""
        return self.headers.copy()