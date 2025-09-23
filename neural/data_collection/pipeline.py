"""
Data pipeline orchestration for the Neural SDK.

This module provides the core data pipeline that coordinates multiple data sources,
transforms raw data, manages buffers, and routes data to consumers. It acts as the
central hub for all data flowing through the system.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import logging

from neural.data_collection.base import BaseDataSource, ConnectionState
from neural.data_collection.exceptions import (
    ConfigurationError,
    DataSourceError,
    BufferOverflowError,
    ValidationError
)


# Configure module logger
logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Standardized data formats for pipeline processing."""
    RAW = "raw"           # Unprocessed data from source
    NORMALIZED = "normalized"  # Standardized format
    ENRICHED = "enriched"     # Data with additional context
    AGGREGATED = "aggregated" # Combined/summarized data


@dataclass
class DataPacket:
    """
    Container for data flowing through the pipeline.
    
    This class wraps data with metadata for tracking and processing
    as it moves through the pipeline stages.
    
    Attributes:
        source_id: Identifier of the data source
        data: The actual data payload
        format: Current data format
        timestamp: When the data was received
        sequence_number: Sequence number for ordering
        metadata: Additional context about the data
        
    Example:
        >>> packet = DataPacket(
        ...     source_id="espn_api",
        ...     data={"team": "Packers", "score": 21},
        ...     format=DataFormat.RAW
        ... )
    """
    source_id: str
    data: Any
    format: DataFormat = DataFormat.RAW
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """Return string representation of the packet."""
        return (
            f"DataPacket(source={self.source_id}, "
            f"format={self.format.value}, "
            f"seq={self.sequence_number})"
        )


class TransformStage:
    """
    Base class for pipeline transformation stages.
    
    Transform stages process data packets as they flow through
    the pipeline, performing operations like normalization,
    enrichment, or filtering.
    
    Subclasses should override the transform method to implement
    specific transformation logic.
    
    Example:
        >>> class NormalizeStage(TransformStage):
        ...     async def transform(self, packet):
        ...         # Normalize the data
        ...         packet.data = normalize(packet.data)
        ...         packet.format = DataFormat.NORMALIZED
        ...         return packet
    """
    
    def __init__(self, name: str):
        """
        Initialize the transform stage.
        
        Args:
            name: Unique name for this stage
        """
        self.name = name
        self.enabled = True
        self.processed_count = 0
        self.error_count = 0
        
    async def transform(self, packet: DataPacket) -> Optional[DataPacket]:
        """
        Transform a data packet.
        
        Args:
            packet: Input data packet
            
        Returns:
            Transformed packet or None to filter out
        """
        # Default implementation passes through unchanged
        return packet
    
    async def process(self, packet: DataPacket) -> Optional[DataPacket]:
        """
        Process a packet through this stage.
        
        Args:
            packet: Input data packet
            
        Returns:
            Processed packet or None if filtered
        """
        if not self.enabled:
            return packet
        
        try:
            result = await self.transform(packet)
            if result:
                self.processed_count += 1
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                f"Transform stage '{self.name}' error: {e}",
                exc_info=True
            )
            # Pass through on error by default
            return packet
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for this stage.
        
        Returns:
            Dictionary with stage statistics
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "processed": self.processed_count,
            "errors": self.error_count
        }


class DataPipeline:
    """
    Central data pipeline orchestrator for the Neural SDK.
    
    This class manages the flow of data from multiple sources through
    transformation stages to multiple consumers. It handles:
    
    - Source registration and lifecycle
    - Data transformation pipeline
    - Consumer/subscriber management
    - Buffer management and backpressure
    - Error handling and recovery
    - Performance monitoring
    
    The pipeline follows this flow:
    Sources → Input Buffer → Transform Stages → Output Buffer → Consumers
    
    Attributes:
        sources: Registered data sources
        stages: Transformation stages in order
        consumers: Registered data consumers
        buffer_size: Maximum buffer size
        
    Example:
        >>> pipeline = DataPipeline(buffer_size=10000)
        >>> 
        >>> # Add a data source
        >>> await pipeline.add_source("ws_source", websocket_source)
        >>> 
        >>> # Add transform stages
        >>> pipeline.add_stage(NormalizeStage("normalize"))
        >>> pipeline.add_stage(EnrichStage("enrich"))
        >>> 
        >>> # Register consumer
        >>> pipeline.subscribe("my_consumer", on_data_callback)
        >>> 
        >>> # Start the pipeline
        >>> await pipeline.start()
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        enable_monitoring: bool = True
    ):
        """
        Initialize the data pipeline.
        
        Args:
            buffer_size: Maximum size for data buffers
            enable_monitoring: Whether to enable performance monitoring
        """
        # Data sources
        self.sources: Dict[str, BaseDataSource] = {}
        self._source_tasks: Dict[str, asyncio.Task] = {}
        
        # Transform stages
        self.stages: List[TransformStage] = []
        
        # Consumers/Subscribers
        self._consumers: Dict[str, List[Callable]] = defaultdict(list)
        self._consumer_filters: Dict[str, Callable] = {}
        
        # Data buffers
        self.buffer_size = buffer_size
        self._input_buffer: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._output_buffer: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        
        # Pipeline state
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._distribute_task: Optional[asyncio.Task] = None
        
        # Monitoring
        self.enable_monitoring = enable_monitoring
        self._metrics = {
            "packets_received": 0,
            "packets_processed": 0,
            "packets_delivered": 0,
            "packets_dropped": 0,
            "processing_time_total": 0.0,
            "start_time": None,
            "errors": 0
        }
        
        # Sequence numbering
        self._sequence_counter = 0
        self._sequence_lock = asyncio.Lock()
        
        logger.info(f"Initialized DataPipeline with buffer_size={buffer_size}")
    
    async def add_source(
        self,
        source_id: str,
        source: BaseDataSource,
        auto_connect: bool = True
    ) -> None:
        """
        Add a data source to the pipeline.
        
        Args:
            source_id: Unique identifier for the source
            source: Data source instance
            auto_connect: Whether to automatically connect the source
            
        Raises:
            ConfigurationError: If source_id already exists
            
        Example:
            >>> await pipeline.add_source("twitter", twitter_source)
        """
        if source_id in self.sources:
            raise ConfigurationError(
                f"Source with id '{source_id}' already exists"
            )
        
        # Register data callback
        async def on_source_data(data: Any):
            """Handle data from source."""
            await self._handle_source_data(source_id, data)
        
        source.register_callback("data", on_source_data)
        
        # Register error callback
        async def on_source_error(error: Any):
            """Handle error from source."""
            logger.error(f"Source '{source_id}' error: {error}")
            self._metrics["errors"] += 1
        
        source.register_callback("error", on_source_error)
        
        # Store source
        self.sources[source_id] = source
        
        # Connect if requested and pipeline is running
        if auto_connect and self._running:
            await self._start_source(source_id)
        
        logger.info(f"Added data source: {source_id}")
    
    async def remove_source(self, source_id: str) -> None:
        """
        Remove a data source from the pipeline.
        
        Args:
            source_id: Identifier of the source to remove
            
        Example:
            >>> await pipeline.remove_source("twitter")
        """
        if source_id not in self.sources:
            logger.warning(f"Source '{source_id}' not found")
            return
        
        # Stop source if running
        await self._stop_source(source_id)
        
        # Remove from registry
        del self.sources[source_id]
        
        logger.info(f"Removed data source: {source_id}")
    
    def add_stage(self, stage: TransformStage) -> None:
        """
        Add a transformation stage to the pipeline.
        
        Stages are executed in the order they are added.
        
        Args:
            stage: Transform stage instance
            
        Example:
            >>> pipeline.add_stage(NormalizeStage("normalize"))
        """
        self.stages.append(stage)
        logger.info(f"Added transform stage: {stage.name}")
    
    def remove_stage(self, stage_name: str) -> None:
        """
        Remove a transformation stage.
        
        Args:
            stage_name: Name of the stage to remove
        """
        self.stages = [s for s in self.stages if s.name != stage_name]
        logger.info(f"Removed transform stage: {stage_name}")
    
    def subscribe(
        self,
        consumer_id: str,
        callback: Callable[[DataPacket], None],
        filter_func: Optional[Callable[[DataPacket], bool]] = None
    ) -> None:
        """
        Subscribe a consumer to receive data packets.
        
        Args:
            consumer_id: Unique identifier for the consumer
            callback: Function to call with data packets
            filter_func: Optional filter to select packets
            
        Example:
            >>> def on_data(packet):
            ...     print(f"Received: {packet.data}")
            >>> 
            >>> # Subscribe to all data
            >>> pipeline.subscribe("printer", on_data)
            >>> 
            >>> # Subscribe with filter
            >>> pipeline.subscribe(
            ...     "espn_only",
            ...     on_data,
            ...     filter_func=lambda p: p.source_id == "espn"
            ... )
        """
        self._consumers[consumer_id].append(callback)
        
        if filter_func:
            self._consumer_filters[consumer_id] = filter_func
        
        logger.info(f"Subscribed consumer: {consumer_id}")
    
    def unsubscribe(self, consumer_id: str) -> None:
        """
        Unsubscribe a consumer.
        
        Args:
            consumer_id: Identifier of the consumer to remove
        """
        if consumer_id in self._consumers:
            del self._consumers[consumer_id]
        
        if consumer_id in self._consumer_filters:
            del self._consumer_filters[consumer_id]
        
        logger.info(f"Unsubscribed consumer: {consumer_id}")
    
    async def _handle_source_data(self, source_id: str, data: Any) -> None:
        """
        Handle data received from a source.
        
        Args:
            source_id: Identifier of the source
            data: Data received from the source
        """
        # Create data packet
        async with self._sequence_lock:
            self._sequence_counter += 1
            sequence_number = self._sequence_counter
        
        packet = DataPacket(
            source_id=source_id,
            data=data,
            format=DataFormat.RAW,
            sequence_number=sequence_number
        )
        
        # Add to input buffer
        try:
            # Try to add without blocking
            self._input_buffer.put_nowait(packet)
            self._metrics["packets_received"] += 1
            
        except asyncio.QueueFull:
            # Buffer full, handle overflow
            self._metrics["packets_dropped"] += 1
            
            # Log warning periodically
            if self._metrics["packets_dropped"] % 100 == 0:
                logger.warning(
                    f"Input buffer overflow, dropped {self._metrics['packets_dropped']} packets"
                )
            
            # Could implement different strategies here:
            # - Drop oldest (would require custom queue)
            # - Block until space available
            # - Increase buffer size dynamically
    
    async def _process_loop(self) -> None:
        """
        Main processing loop that transforms data packets.
        
        This coroutine continuously processes packets from the input
        buffer through the transform stages to the output buffer.
        """
        logger.debug("Starting pipeline processing loop")
        
        while self._running:
            try:
                # Get packet from input buffer with timeout
                packet = await asyncio.wait_for(
                    self._input_buffer.get(),
                    timeout=1.0
                )
                
                start_time = time.time()
                
                # Process through transform stages
                for stage in self.stages:
                    if packet is None:
                        break
                    packet = await stage.process(packet)
                
                # Add to output buffer if not filtered
                if packet:
                    try:
                        await self._output_buffer.put(packet)
                        self._metrics["packets_processed"] += 1
                        
                        # Track processing time
                        processing_time = time.time() - start_time
                        self._metrics["processing_time_total"] += processing_time
                        
                    except asyncio.QueueFull:
                        self._metrics["packets_dropped"] += 1
                        logger.warning("Output buffer full, dropping packet")
                
            except asyncio.TimeoutError:
                # No data available, continue
                continue
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}", exc_info=True)
                self._metrics["errors"] += 1
        
        logger.debug("Pipeline processing loop stopped")
    
    async def _distribute_loop(self) -> None:
        """
        Distribution loop that sends packets to consumers.
        
        This coroutine continuously delivers packets from the output
        buffer to registered consumers.
        """
        logger.debug("Starting pipeline distribution loop")
        
        while self._running:
            try:
                # Get packet from output buffer with timeout
                packet = await asyncio.wait_for(
                    self._output_buffer.get(),
                    timeout=1.0
                )
                
                # Distribute to consumers
                for consumer_id, callbacks in self._consumers.items():
                    # Check filter if exists
                    if consumer_id in self._consumer_filters:
                        filter_func = self._consumer_filters[consumer_id]
                        if not filter_func(packet):
                            continue
                    
                    # Call all callbacks for this consumer
                    for callback in callbacks:
                        try:
                            # Handle both sync and async callbacks
                            if asyncio.iscoroutinefunction(callback):
                                await callback(packet)
                            else:
                                callback(packet)
                            
                            self._metrics["packets_delivered"] += 1
                            
                        except Exception as e:
                            logger.error(
                                f"Consumer '{consumer_id}' callback error: {e}",
                                exc_info=True
                            )
                            self._metrics["errors"] += 1
                
            except asyncio.TimeoutError:
                # No data available, continue
                continue
                
            except Exception as e:
                logger.error(f"Distribution loop error: {e}", exc_info=True)
                self._metrics["errors"] += 1
        
        logger.debug("Pipeline distribution loop stopped")
    
    async def _start_source(self, source_id: str) -> None:
        """
        Start a single data source.
        
        Args:
            source_id: Identifier of the source to start
        """
        source = self.sources[source_id]
        
        if not source.is_connected():
            try:
                await source.connect()
                logger.info(f"Connected source: {source_id}")
            except Exception as e:
                logger.error(f"Failed to connect source '{source_id}': {e}")
                raise
    
    async def _stop_source(self, source_id: str) -> None:
        """
        Stop a single data source.
        
        Args:
            source_id: Identifier of the source to stop
        """
        if source_id in self.sources:
            source = self.sources[source_id]
            
            if source.is_connected():
                try:
                    await source.disconnect()
                    logger.info(f"Disconnected source: {source_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting source '{source_id}': {e}")
    
    async def start(self) -> None:
        """
        Start the data pipeline.
        
        This method starts all registered sources and begins
        processing data through the pipeline.
        
        Example:
            >>> await pipeline.start()
        """
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting data pipeline")
        
        # Set running flag
        self._running = True
        self._metrics["start_time"] = datetime.now()
        
        # Start all sources
        for source_id in self.sources:
            try:
                await self._start_source(source_id)
            except Exception as e:
                logger.error(f"Failed to start source '{source_id}': {e}")
                # Continue with other sources
        
        # Start processing loops
        self._process_task = asyncio.create_task(
            self._process_loop(),
            name="pipeline_process"
        )
        
        self._distribute_task = asyncio.create_task(
            self._distribute_loop(),
            name="pipeline_distribute"
        )
        
        logger.info("Data pipeline started successfully")
    
    async def stop(self) -> None:
        """
        Stop the data pipeline.
        
        This method stops all sources and processing loops,
        allowing buffered data to be processed before shutdown.
        
        Example:
            >>> await pipeline.stop()
        """
        if not self._running:
            logger.warning("Pipeline not running")
            return
        
        logger.info("Stopping data pipeline")
        
        # Set running flag
        self._running = False
        
        # Stop all sources
        for source_id in self.sources:
            await self._stop_source(source_id)
        
        # Wait for processing tasks to complete
        if self._process_task:
            try:
                await asyncio.wait_for(self._process_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._process_task.cancel()
        
        if self._distribute_task:
            try:
                await asyncio.wait_for(self._distribute_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._distribute_task.cancel()
        
        logger.info("Data pipeline stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline performance metrics.
        
        Returns:
            Dictionary containing pipeline metrics
            
        Example:
            >>> metrics = pipeline.get_metrics()
            >>> print(f"Processed: {metrics['packets_processed']}")
        """
        metrics = self._metrics.copy()
        
        # Calculate rates if pipeline has been running
        if metrics["start_time"]:
            runtime = (datetime.now() - metrics["start_time"]).total_seconds()
            if runtime > 0:
                metrics["receive_rate"] = metrics["packets_received"] / runtime
                metrics["process_rate"] = metrics["packets_processed"] / runtime
                metrics["delivery_rate"] = metrics["packets_delivered"] / runtime
        
        # Add stage statistics
        metrics["stages"] = [stage.get_stats() for stage in self.stages]
        
        # Add source statistics
        metrics["sources"] = {
            source_id: source.get_metrics()
            for source_id, source in self.sources.items()
        }
        
        # Buffer status
        metrics["input_buffer_size"] = self._input_buffer.qsize()
        metrics["output_buffer_size"] = self._output_buffer.qsize()
        
        return metrics
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            "running": self._running,
            "sources": list(self.sources.keys()),
            "stages": [stage.name for stage in self.stages],
            "consumers": list(self._consumers.keys()),
            "buffer_usage": {
                "input": f"{self._input_buffer.qsize()}/{self.buffer_size}",
                "output": f"{self._output_buffer.qsize()}/{self.buffer_size}"
            }
        }
    
    async def flush(self) -> None:
        """
        Flush all buffered data through the pipeline.
        
        This method processes all data currently in buffers
        without stopping the pipeline.
        
        Example:
            >>> await pipeline.flush()
        """
        logger.info("Flushing pipeline buffers")
        
        # Wait for input buffer to empty
        while not self._input_buffer.empty():
            await asyncio.sleep(0.1)
        
        # Wait for output buffer to empty
        while not self._output_buffer.empty():
            await asyncio.sleep(0.1)
        
        logger.info("Pipeline buffers flushed")
    
    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return (
            f"DataPipeline("
            f"sources={len(self.sources)}, "
            f"stages={len(self.stages)}, "
            f"consumers={len(self._consumers)}, "
            f"running={self._running})"
        )