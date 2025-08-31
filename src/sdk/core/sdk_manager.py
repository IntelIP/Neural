"""
SDK Manager for Neural Trading Platform
Handles adapter discovery, loading, and orchestration
"""

import asyncio
import importlib
import importlib.util
import sys
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Type, AsyncGenerator, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

from .base_adapter import DataSourceAdapter, StandardizedEvent, DataSourceMetadata

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for a data source adapter"""
    name: str
    enabled: bool
    class_name: str
    module_path: str
    config: Dict[str, Any]
    priority: int = 0  # Higher priority sources get processed first
    

class SDKManager:
    """
    Central manager for all data source adapters
    Handles discovery, loading, and orchestration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize SDK Manager
        
        Args:
            config_path: Path to configuration file
        """
        self.adapters: Dict[str, DataSourceAdapter] = {}
        self.adapter_configs: Dict[str, AdapterConfig] = {}
        self.config_path = config_path or "config/data_sources.yaml"
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._event_queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("SDK Manager initialized")
    
    def discover_adapters(self, adapter_dir: str = "src/sdk/adapters") -> List[str]:
        """
        Discover available adapters in directory
        
        Args:
            adapter_dir: Directory to search for adapters
            
        Returns:
            List of discovered adapter names
        """
        adapter_path = Path(adapter_dir)
        discovered = []
        
        if not adapter_path.exists():
            logger.warning(f"Adapter directory does not exist: {adapter_dir}")
            return discovered
        
        for file_path in adapter_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            # Try to load and validate the module
            module_name = file_path.stem
            try:
                spec = importlib.util.spec_from_file_location(
                    f"adapters.{module_name}", 
                    file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find adapter classes
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (isinstance(item, type) and 
                        issubclass(item, DataSourceAdapter) and 
                        item != DataSourceAdapter):
                        discovered.append(module_name)
                        logger.info(f"Discovered adapter: {module_name}")
                        break
                        
            except Exception as e:
                logger.error(f"Error discovering adapter {module_name}: {e}")
        
        return discovered
    
    def load_config(self) -> Dict[str, AdapterConfig]:
        """
        Load adapter configuration from YAML file
        
        Returns:
            Dictionary of adapter configurations
        """
        configs = {}
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for source in config_data.get('sources', []):
                # Handle environment variables in config
                self._substitute_env_vars(source.get('config', {}))
                
                adapter_config = AdapterConfig(
                    name=source['name'],
                    enabled=source.get('enabled', True),
                    class_name=source.get('class', f"{source['name'].title()}Adapter"),
                    module_path=source.get('module', f"src.sdk.adapters.{source['name']}"),
                    config=source.get('config', {}),
                    priority=source.get('priority', 0)
                )
                
                configs[source['name']] = adapter_config
                
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return configs
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> None:
        """
        Substitute environment variables in configuration
        Format: ${ENV_VAR_NAME}
        
        Args:
            config: Configuration dictionary to modify in place
        """
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, value)
            elif isinstance(value, dict):
                self._substitute_env_vars(value)
    
    async def load_adapter(self, adapter_config: AdapterConfig) -> Optional[DataSourceAdapter]:
        """
        Load and initialize a single adapter
        
        Args:
            adapter_config: Adapter configuration
            
        Returns:
            Initialized adapter or None if loading fails
        """
        try:
            # Import the module
            module = importlib.import_module(adapter_config.module_path)
            
            # Get the adapter class
            adapter_class = getattr(module, adapter_config.class_name)
            
            # Validate it's a proper adapter
            if not issubclass(adapter_class, DataSourceAdapter):
                raise ValueError(f"{adapter_config.class_name} is not a DataSourceAdapter")
            
            # Instantiate the adapter
            adapter = adapter_class(adapter_config.config)
            
            # Start the adapter
            await adapter.start()
            
            logger.info(f"Loaded adapter: {adapter_config.name}")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_config.name}: {e}")
            return None
    
    async def initialize(self) -> None:
        """
        Initialize all configured adapters
        """
        # Load configuration
        self.adapter_configs = self.load_config()
        
        # Load enabled adapters
        for name, config in self.adapter_configs.items():
            if not config.enabled:
                logger.info(f"Skipping disabled adapter: {name}")
                continue
            
            adapter = await self.load_adapter(config)
            if adapter:
                self.adapters[name] = adapter
        
        logger.info(f"Initialized {len(self.adapters)} adapters")
    
    async def start(self) -> None:
        """
        Start all adapters and begin streaming
        """
        if self._running:
            logger.warning("SDK Manager already running")
            return
        
        self._running = True
        
        # Start streaming from each adapter
        for name, adapter in self.adapters.items():
            task = asyncio.create_task(
                self._stream_adapter(name, adapter)
            )
            self._tasks.append(task)
        
        # Start event processor
        processor_task = asyncio.create_task(self._process_events())
        self._tasks.append(processor_task)
        
        logger.info(f"Started streaming from {len(self.adapters)} adapters")
    
    async def stop(self) -> None:
        """
        Stop all adapters and cleanup
        """
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop all adapters
        for adapter in self.adapters.values():
            await adapter.stop()
        
        self._tasks.clear()
        logger.info("SDK Manager stopped")
    
    async def _stream_adapter(self, name: str, adapter: DataSourceAdapter) -> None:
        """
        Stream events from a single adapter
        
        Args:
            name: Adapter name
            adapter: Adapter instance
        """
        while self._running:
            try:
                async for event in adapter.stream():
                    # Add adapter name to event
                    event.source = name
                    
                    # Put event in queue for processing
                    await self._event_queue.put(event)
                    
                    # Update adapter statistics
                    adapter._increment_event_count()
                    
            except Exception as e:
                logger.error(f"Error streaming from {name}: {e}")
                adapter._increment_error_count()
                
                # Reconnect after error
                if self._running:
                    await asyncio.sleep(5)
                    try:
                        await adapter.start()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect {name}: {reconnect_error}")
    
    async def _process_events(self) -> None:
        """
        Process events from all adapters
        """
        while self._running:
            try:
                # Get event with timeout to allow checking _running flag
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                # Process event (this is where you'd send to trading logic)
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: StandardizedEvent) -> None:
        """
        Handle a standardized event
        
        Args:
            event: Event to process
        """
        # Log high-impact events
        if event.impact in ["high", "critical"]:
            logger.info(
                f"High impact event from {event.source}: "
                f"{event.event_type.value} (confidence: {event.confidence:.2f})"
            )
        
        # This is where events would be sent to trading strategies
        # For now, just track them
        pass
    
    async def get_events(self) -> AsyncGenerator[StandardizedEvent, None]:
        """
        Get standardized events from all adapters
        
        Yields:
            StandardizedEvent objects from all sources
        """
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                yield event
            except asyncio.TimeoutError:
                continue
    
    def get_adapter(self, name: str) -> Optional[DataSourceAdapter]:
        """
        Get a specific adapter by name
        
        Args:
            name: Adapter name
            
        Returns:
            Adapter instance or None
        """
        return self.adapters.get(name)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all adapters
        
        Returns:
            Health status for all adapters
        """
        health_status = {
            "manager_running": self._running,
            "adapters": {}
        }
        
        for name, adapter in self.adapters.items():
            health_status["adapters"][name] = await adapter.health_check()
        
        return health_status
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all adapters
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_adapters": len(self.adapters),
            "running": self._running,
            "queue_size": self._event_queue.qsize(),
            "adapters": {}
        }
        
        for name, adapter in self.adapters.items():
            stats["adapters"][name] = adapter.statistics
        
        return stats
    
    async def test_adapter(self, name: str, duration: int = 60) -> Dict[str, Any]:
        """
        Test a specific adapter
        
        Args:
            name: Adapter name
            duration: Test duration in seconds
            
        Returns:
            Test results
        """
        adapter = self.adapters.get(name)
        if not adapter:
            return {"error": f"Adapter {name} not found"}
        
        results = {
            "adapter": name,
            "duration": duration,
            "events_received": 0,
            "errors": 0,
            "latency_samples": [],
            "event_types": {}
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async for event in adapter.stream():
                results["events_received"] += 1
                
                # Track event types
                event_type = event.event_type.value
                results["event_types"][event_type] = results["event_types"].get(event_type, 0) + 1
                
                # Sample latency
                if results["events_received"] % 10 == 0:
                    latency = (datetime.now() - event.timestamp).total_seconds() * 1000
                    results["latency_samples"].append(latency)
                
                # Check if duration exceeded
                if asyncio.get_event_loop().time() - start_time > duration:
                    break
                    
        except Exception as e:
            results["errors"] += 1
            results["error_message"] = str(e)
        
        # Calculate statistics
        if results["latency_samples"]:
            results["avg_latency_ms"] = sum(results["latency_samples"]) / len(results["latency_samples"])
            results["max_latency_ms"] = max(results["latency_samples"])
            results["min_latency_ms"] = min(results["latency_samples"])
        
        results["events_per_second"] = results["events_received"] / duration
        
        return results