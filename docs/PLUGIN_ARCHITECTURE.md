# Plugin Architecture Design

## Overview
Enable users to extend the Neural Trading Platform with custom data sources, strategies, and indicators without modifying core code.

## Core Components

### 1. Plugin Base Classes

```python
# src/plugins/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class PluginMetadata:
    """Plugin identification and requirements"""
    name: str
    version: str
    author: str
    description: str
    requirements: List[str]
    config_schema: Dict[str, Any]

class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata = self.get_metadata()
        self.validate_config()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate plugin configuration"""
        pass
    
    @abstractmethod
    async def initialize(self):
        """Initialize plugin resources"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean up plugin resources"""
        pass
```

### 2. Data Source Plugins

```python
# src/plugins/data_source.py
from typing import AsyncGenerator, Optional
from datetime import datetime

@dataclass
class MarketData:
    """Standardized market data format"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataSourcePlugin(BasePlugin):
    """Base class for data source plugins"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to market data for symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data"""
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """Stream market data"""
        pass
    
    @abstractmethod
    def transform(self, raw_data: Any) -> MarketData:
        """Transform raw data to standardized format"""
        pass
```

### 3. Strategy Plugins

```python
# src/plugins/strategy.py
from enum import Enum
from typing import Optional

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyPlugin(BasePlugin):
    """Base class for trading strategy plugins"""
    
    @abstractmethod
    def analyze(self, data: MarketData, history: List[MarketData]) -> Optional[TradingSignal]:
        """Analyze market data and generate signal"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, capital: float) -> float:
        """Calculate position size for signal"""
        pass
    
    @abstractmethod
    def update_state(self, data: MarketData):
        """Update internal strategy state"""
        pass
    
    @abstractmethod
    def get_indicators(self) -> Dict[str, float]:
        """Get current indicator values"""
        pass
```

### 4. Plugin Manager

```python
# src/plugins/manager.py
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Type

class PluginManager:
    """Manages plugin lifecycle and registration"""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, BasePlugin] = {}
        self.data_sources: Dict[str, DataSourcePlugin] = {}
        self.strategies: Dict[str, StrategyPlugin] = {}
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directory"""
        plugins = []
        for file in self.plugin_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue
            plugins.append(file.stem)
        return plugins
    
    def load_plugin(self, plugin_name: str, config: Dict[str, Any]) -> BasePlugin:
        """Load a plugin from file"""
        plugin_path = self.plugin_dir / f"{plugin_name}.py"
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_name] = module
        spec.loader.exec_module(module)
        
        # Find plugin class
        plugin_class = None
        for item in dir(module):
            obj = getattr(module, item)
            if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj != BasePlugin:
                plugin_class = obj
                break
        
        if not plugin_class:
            raise ValueError(f"No plugin class found in {plugin_name}")
        
        # Instantiate plugin
        plugin = plugin_class(config)
        
        # Register by type
        if isinstance(plugin, DataSourcePlugin):
            self.data_sources[plugin_name] = plugin
        elif isinstance(plugin, StrategyPlugin):
            self.strategies[plugin_name] = plugin
        
        self.plugins[plugin_name] = plugin
        return plugin
    
    async def initialize_all(self):
        """Initialize all loaded plugins"""
        for plugin in self.plugins.values():
            await plugin.initialize()
    
    async def shutdown_all(self):
        """Shutdown all plugins"""
        for plugin in self.plugins.values():
            await plugin.shutdown()
```

## Example Plugins

### Custom WebSocket Data Source

```python
# plugins/binance_data.py
import websockets
import json
from datetime import datetime
from typing import AsyncGenerator, List
from src.plugins.data_source import DataSourcePlugin, MarketData
from src.plugins.base import PluginMetadata

class BinanceDataPlugin(DataSourcePlugin):
    """Binance WebSocket data source plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Binance Data",
            version="1.0.0",
            author="User",
            description="Real-time Binance market data",
            requirements=["websockets>=10.0"],
            config_schema={
                "stream_url": "str",
                "symbols": "list"
            }
        )
    
    async def connect(self) -> bool:
        """Connect to Binance WebSocket"""
        self.ws = await websockets.connect(
            "wss://stream.binance.com:9443/ws"
        )
        return True
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbol streams"""
        streams = [f"{s.lower()}@trade" for s in symbols]
        await self.ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }))
        return True
    
    async def stream(self) -> AsyncGenerator[MarketData, None]:
        """Stream market data"""
        async for message in self.ws:
            data = json.loads(message)
            if 'e' in data and data['e'] == 'trade':
                yield self.transform(data)
    
    def transform(self, raw_data: dict) -> MarketData:
        """Transform Binance data to standard format"""
        return MarketData(
            timestamp=datetime.fromtimestamp(raw_data['T'] / 1000),
            symbol=raw_data['s'],
            price=float(raw_data['p']),
            volume=float(raw_data['q'])
        )
```

### Custom Trading Strategy

```python
# plugins/bollinger_strategy.py
import numpy as np
from typing import Optional, List
from src.plugins.strategy import StrategyPlugin, TradingSignal, SignalType
from src.plugins.base import PluginMetadata

class BollingerBandsStrategy(StrategyPlugin):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, config):
        super().__init__(config)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2)
        self.prices = []
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Bollinger Bands",
            version="1.0.0",
            author="User",
            description="Mean reversion using Bollinger Bands",
            requirements=["numpy>=1.20"],
            config_schema={
                "period": "int",
                "std_dev": "float",
                "position_size": "float"
            }
        )
    
    def analyze(self, data: MarketData, history: List[MarketData]) -> Optional[TradingSignal]:
        """Generate trading signal"""
        # Update price history
        self.prices.append(data.price)
        if len(self.prices) > self.period:
            self.prices.pop(0)
        
        if len(self.prices) < self.period:
            return None
        
        # Calculate Bollinger Bands
        sma = np.mean(self.prices)
        std = np.std(self.prices)
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        # Generate signals
        if data.price < lower_band:
            return TradingSignal(
                symbol=data.symbol,
                signal_type=SignalType.BUY,
                confidence=min(1.0, (sma - data.price) / std),
                price=data.price,
                take_profit=sma,
                stop_loss=lower_band - std
            )
        elif data.price > upper_band:
            return TradingSignal(
                symbol=data.symbol,
                signal_type=SignalType.SELL,
                confidence=min(1.0, (data.price - sma) / std),
                price=data.price,
                take_profit=sma,
                stop_loss=upper_band + std
            )
        
        return None
    
    def calculate_position_size(self, signal: TradingSignal, capital: float) -> float:
        """Kelly Criterion position sizing"""
        kelly_fraction = signal.confidence * 0.25  # Conservative Kelly
        return capital * kelly_fraction * self.config.get('position_size', 0.02)
```

## Plugin Configuration

### YAML Configuration

```yaml
# config/plugins.yaml
plugins:
  data_sources:
    - name: binance_data
      enabled: true
      config:
        stream_url: "wss://stream.binance.com:9443/ws"
        symbols: ["BTCUSDT", "ETHUSDT"]
    
    - name: polygon_data
      enabled: true
      config:
        api_key: "${POLYGON_API_KEY}"
        symbols: ["AAPL", "GOOGL"]
  
  strategies:
    - name: bollinger_strategy
      enabled: true
      config:
        period: 20
        std_dev: 2.0
        position_size: 0.02
    
    - name: momentum_strategy
      enabled: true
      config:
        lookback: 14
        threshold: 0.7
```

### Environment Variables

```bash
# .env
PLUGIN_DIR=./plugins
PLUGIN_CONFIG=./config/plugins.yaml
PLUGIN_SANDBOX=true
PLUGIN_MAX_MEMORY=512MB
PLUGIN_TIMEOUT=30
```

## Security & Sandboxing

### Plugin Sandbox

```python
# src/plugins/sandbox.py
import resource
import signal
from contextlib import contextmanager

class PluginSandbox:
    """Sandbox for plugin execution"""
    
    def __init__(self, max_memory_mb: int = 512, timeout_seconds: int = 30):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.timeout = timeout_seconds
    
    @contextmanager
    def sandbox(self):
        """Create sandboxed environment"""
        # Set memory limit
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.max_memory, self.max_memory)
        )
        
        # Set CPU time limit
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            yield
        finally:
            signal.alarm(0)
    
    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Plugin execution timeout")
```

## Plugin Development SDK

### CLI Tool

```bash
# Neural Trading Platform Plugin CLI
ntp plugin create my_strategy --template bollinger
ntp plugin test my_strategy --data sample_data.csv
ntp plugin validate my_strategy
ntp plugin package my_strategy
ntp plugin publish my_strategy --marketplace
```

### Testing Framework

```python
# src/plugins/testing.py
class PluginTester:
    """Test framework for plugins"""
    
    async def test_data_source(self, plugin: DataSourcePlugin):
        """Test data source plugin"""
        # Test connection
        assert await plugin.connect()
        
        # Test subscription
        assert await plugin.subscribe(["TEST"])
        
        # Test streaming
        async for data in plugin.stream():
            assert isinstance(data, MarketData)
            break
        
        # Test disconnection
        await plugin.disconnect()
    
    async def test_strategy(self, plugin: StrategyPlugin, test_data: List[MarketData]):
        """Test strategy plugin"""
        signals = []
        for data in test_data:
            signal = plugin.analyze(data, test_data[:test_data.index(data)])
            if signal:
                signals.append(signal)
        
        # Validate signals
        for signal in signals:
            assert 0 <= signal.confidence <= 1
            assert signal.signal_type in SignalType
```

## Integration with Main Platform

```python
# src/main.py
async def main():
    # Initialize plugin manager
    plugin_manager = PluginManager("./plugins")
    
    # Discover and load plugins
    plugins = plugin_manager.discover_plugins()
    for plugin_name in plugins:
        config = load_plugin_config(plugin_name)
        plugin_manager.load_plugin(plugin_name, config)
    
    # Initialize all plugins
    await plugin_manager.initialize_all()
    
    # Create unified stream from all data sources
    async def unified_stream():
        for name, source in plugin_manager.data_sources.items():
            async for data in source.stream():
                # Process with all strategies
                for strategy_name, strategy in plugin_manager.strategies.items():
                    signal = strategy.analyze(data, [])
                    if signal:
                        await process_signal(signal)
    
    # Run main loop
    await unified_stream()
```

This plugin architecture provides:
1. **Extensibility** - Users can add any data source or strategy
2. **Safety** - Sandboxed execution prevents system damage
3. **Simplicity** - Clear interfaces and examples
4. **Performance** - Async throughout for efficiency
5. **Marketplace Ready** - Built for sharing and monetization