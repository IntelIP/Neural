#!/usr/bin/env python3
"""
OpenAPI specification generator for Neural SDK.
Generates OpenAPI specs from REST API endpoints and data models.
"""

import json
from pathlib import Path
from typing import Any


class OpenAPIGenerator:
    def __init__(self, output_dir: Path = Path("docs/openapi")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Neural SDK API",
                "version": "0.3.0",
                "description": "REST API for Neural SDK trading and data collection functionality",
                "contact": {"name": "Neural SDK Team", "email": "support@neural-sdk.com"},
                "license": {
                    "name": "MIT",
                    "url": "https://github.com/IntelIP/Neural/blob/main/LICENSE",
                },
            },
            "servers": [
                {"url": "https://api.kalshi.com", "description": "Production server"},
                {"url": "https://demo-api.kalshi.com", "description": "Demo server"},
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "Authorization"}
                },
            },
            "security": [{"ApiKeyAuth": []}],
        }

    def generate_all(self) -> bool:
        """Generate all OpenAPI specifications."""
        print("🔧 Generating OpenAPI specifications...")

        try:
            # Generate trading API specs
            self._generate_trading_specs()

            # Generate data collection API specs
            self._generate_data_collection_specs()

            # Generate authentication API specs
            self._generate_auth_specs()

            # Save the main specification
            self._save_specification("neural-sdk-api.json", self.spec)

            # Generate separate specs for different modules
            self._generate_module_specs()

            print("✅ OpenAPI specifications generated successfully")
            return True

        except Exception as e:
            print(f"❌ Error generating OpenAPI specs: {e}")
            return False

    def _generate_trading_specs(self) -> None:
        """Generate trading API specifications."""
        trading_paths = {
            "/trading/orders": {
                "get": {
                    "summary": "List orders",
                    "description": "Retrieve a list of user orders with optional filtering",
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["open", "filled", "cancelled"]},
                            "description": "Filter by order status",
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer", "default": 100},
                            "description": "Maximum number of orders to return",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "List of orders",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "orders": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Order"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                },
                "post": {
                    "summary": "Place a new order",
                    "description": "Submit a new order to the trading platform",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PlaceOrderRequest"}
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Order placed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/OrderResponse"}
                                }
                            },
                        },
                        "400": {"description": "Invalid order parameters"},
                    },
                },
            },
            "/trading/orders/{order_id}": {
                "get": {
                    "summary": "Get order details",
                    "description": "Retrieve detailed information about a specific order",
                    "parameters": [
                        {
                            "name": "order_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Unique identifier for the order",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Order details",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Order"}
                                }
                            },
                        },
                        "404": {"description": "Order not found"},
                    },
                },
                "delete": {
                    "summary": "Cancel order",
                    "description": "Cancel a pending order",
                    "parameters": [
                        {
                            "name": "order_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "Order cancelled successfully"},
                        "404": {"description": "Order not found"},
                    },
                },
            },
            "/trading/positions": {
                "get": {
                    "summary": "List positions",
                    "description": "Retrieve current trading positions",
                    "responses": {
                        "200": {
                            "description": "List of positions",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "positions": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Position"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/trading/portfolio": {
                "get": {
                    "summary": "Get portfolio summary",
                    "description": "Retrieve portfolio overview including balance and P&L",
                    "responses": {
                        "200": {
                            "description": "Portfolio summary",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Portfolio"}
                                }
                            },
                        }
                    },
                }
            },
        }

        self.spec["paths"].update(trading_paths)

    def _generate_data_collection_specs(self) -> None:
        """Generate data collection API specifications."""
        data_paths = {
            "/data/markets": {
                "get": {
                    "summary": "List available markets",
                    "description": "Retrieve list of available trading markets",
                    "parameters": [
                        {
                            "name": "event_ticker",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Filter by event ticker",
                        },
                        {
                            "name": "category",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Filter by market category",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "List of markets",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "markets": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Market"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/data/markets/{market_id}/price": {
                "get": {
                    "summary": "Get market price",
                    "description": "Retrieve current price for a specific market",
                    "parameters": [
                        {
                            "name": "market_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Market price data",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MarketPrice"}
                                }
                            },
                        }
                    },
                }
            },
            "/data/historical": {
                "get": {
                    "summary": "Get historical data",
                    "description": "Retrieve historical market data",
                    "parameters": [
                        {
                            "name": "market_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "start_date",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "format": "date"},
                        },
                        {
                            "name": "end_date",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "format": "date"},
                        },
                        {
                            "name": "granularity",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["1m", "5m", "1h", "1d"]},
                            "description": "Data granularity",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Historical data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "data": {
                                                "type": "array",
                                                "items": {
                                                    "$ref": "#/components/schemas/HistoricalDataPoint"
                                                },
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
        }

        self.spec["paths"].update(data_paths)

    def _generate_auth_specs(self) -> None:
        """Generate authentication API specifications."""
        auth_paths = {
            "/auth/login": {
                "post": {
                    "summary": "User login",
                    "description": "Authenticate user and obtain access token",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LoginRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/LoginResponse"}
                                }
                            },
                        },
                        "401": {"description": "Invalid credentials"},
                    },
                }
            },
            "/auth/refresh": {
                "post": {
                    "summary": "Refresh access token",
                    "description": "Refresh an expired access token",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RefreshTokenRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Token refreshed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RefreshTokenResponse"}
                                }
                            },
                        }
                    },
                }
            },
        }

        self.spec["paths"].update(auth_paths)

    def _generate_schemas(self) -> None:
        """Generate component schemas."""
        schemas = {
            "Order": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique order identifier"},
                    "market_id": {"type": "string", "description": "Market identifier"},
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Order side",
                    },
                    "quantity": {"type": "integer", "description": "Order quantity"},
                    "price": {"type": "number", "description": "Order price"},
                    "status": {
                        "type": "string",
                        "enum": ["open", "filled", "cancelled"],
                        "description": "Order status",
                    },
                    "created_at": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Order creation time",
                    },
                    "updated_at": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Last update time",
                    },
                },
                "required": ["id", "market_id", "side", "quantity", "price", "status"],
            },
            "PlaceOrderRequest": {
                "type": "object",
                "properties": {
                    "market_id": {"type": "string"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "quantity": {"type": "integer"},
                    "price": {"type": "number"},
                    "order_type": {
                        "type": "string",
                        "enum": ["limit", "market"],
                        "default": "limit",
                    },
                },
                "required": ["market_id", "side", "quantity"],
            },
            "OrderResponse": {
                "type": "object",
                "properties": {
                    "order": {"$ref": "#/components/schemas/Order"},
                    "message": {"type": "string"},
                },
            },
            "Position": {
                "type": "object",
                "properties": {
                    "market_id": {"type": "string"},
                    "side": {"type": "string", "enum": ["long", "short"]},
                    "quantity": {"type": "integer"},
                    "average_price": {"type": "number"},
                    "current_price": {"type": "number"},
                    "unrealized_pnl": {"type": "number"},
                    "realized_pnl": {"type": "number"},
                },
            },
            "Portfolio": {
                "type": "object",
                "properties": {
                    "total_balance": {"type": "number"},
                    "available_balance": {"type": "number"},
                    "total_pnl": {"type": "number"},
                    "positions_count": {"type": "integer"},
                    "orders_count": {"type": "integer"},
                },
            },
            "Market": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "event_ticker": {"type": "string"},
                    "title": {"type": "string"},
                    "category": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "closed", "settled"]},
                    "settlement_time": {"type": "string", "format": "date-time"},
                    "yes_price": {"type": "number"},
                    "no_price": {"type": "number"},
                },
            },
            "MarketPrice": {
                "type": "object",
                "properties": {
                    "market_id": {"type": "string"},
                    "price": {"type": "number"},
                    "volume": {"type": "integer"},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
            "HistoricalDataPoint": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "open": {"type": "number"},
                    "high": {"type": "number"},
                    "low": {"type": "number"},
                    "close": {"type": "number"},
                    "volume": {"type": "integer"},
                },
            },
            "LoginRequest": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "password": {"type": "string"},
                },
                "required": ["email", "password"],
            },
            "LoginResponse": {
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "refresh_token": {"type": "string"},
                    "expires_in": {"type": "integer"},
                    "user": {"$ref": "#/components/schemas/User"},
                },
            },
            "RefreshTokenRequest": {
                "type": "object",
                "properties": {"refresh_token": {"type": "string"}},
                "required": ["refresh_token"],
            },
            "RefreshTokenResponse": {
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "expires_in": {"type": "integer"},
                },
            },
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "email": {"type": "string"},
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                },
            },
        }

        self.spec["components"]["schemas"].update(schemas)

    def _save_specification(self, filename: str, spec: dict[str, Any]) -> None:
        """Save OpenAPI specification to file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(spec, f, indent=2)

    def _generate_module_specs(self) -> None:
        """Generate separate specifications for different modules."""
        # Trading API spec
        trading_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Neural SDK Trading API",
                "version": "0.3.0",
                "description": "Trading and order management API",
            },
            "servers": self.spec["servers"],
            "paths": {},
            "components": self.spec["components"],
        }

        # Filter trading paths
        trading_paths = {k: v for k, v in self.spec["paths"].items() if k.startswith("/trading")}
        trading_spec["paths"] = trading_paths

        self._save_specification("trading-api.json", trading_spec)

        # Data collection API spec
        data_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Neural SDK Data Collection API",
                "version": "0.3.0",
                "description": "Market data and historical data API",
            },
            "servers": self.spec["servers"],
            "paths": {},
            "components": self.spec["components"],
        }

        # Filter data paths
        data_paths = {k: v for k, v in self.spec["paths"].items() if k.startswith("/data")}
        data_spec["paths"] = data_paths

        self._save_specification("data-collection-api.json", data_spec)

        # Auth API spec
        auth_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Neural SDK Authentication API",
                "version": "0.3.0",
                "description": "User authentication and authorization API",
            },
            "servers": self.spec["servers"],
            "paths": {},
            "components": self.spec["components"],
        }

        # Filter auth paths
        auth_paths = {k: v for k, v in self.spec["paths"].items() if k.startswith("/auth")}
        auth_spec["paths"] = auth_paths

        self._save_specification("auth-api.json", auth_spec)


if __name__ == "__main__":
    generator = OpenAPIGenerator()
    success = generator.generate_all()
    exit(0 if success else 1)
