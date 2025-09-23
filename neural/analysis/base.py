"""
Base classes and abstractions for Neural SDK Analysis Infrastructure.

This module provides abstract base classes that define common interfaces
for analysis components, ensuring consistency across different implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    EDGE_DETECTION = "edge_detection"
    PROBABILITY_ESTIMATION = "probability_estimation"
    CORRELATION_ANALYSIS = "correlation_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"


class SignalStrength(Enum):
    """Signal strength levels for trading decisions."""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0


@dataclass
class AnalysisConfig:
    """
    Configuration for analysis operations.
    
    This dataclass defines common configuration parameters
    that all analysis components should support.
    """
    # Time window for analysis
    lookback_hours: int = 24
    min_data_points: int = 10
    
    # Confidence thresholds
    confidence_level: float = 0.95
    min_edge_threshold: float = 0.03
    
    # Risk parameters
    max_position_size: float = 0.10
    use_kelly_sizing: bool = True
    
    # Data sources
    use_sportsbook_data: bool = True
    use_social_sentiment: bool = False
    
    # Additional parameters
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.confidence_level <= 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {self.confidence_level}")
        
        if self.min_edge_threshold < 0:
            raise ValueError(f"Min edge threshold must be non-negative, got {self.min_edge_threshold}")
        
        if not 0 < self.max_position_size <= 1:
            raise ValueError(f"Max position size must be between 0 and 1, got {self.max_position_size}")


@dataclass
class AnalysisResult:
    """
    Result of an analysis operation.
    
    This dataclass encapsulates the output of any analysis,
    providing a consistent interface for results.
    """
    # Core results
    analysis_type: AnalysisType
    timestamp: datetime
    market_id: str
    
    # Primary output
    value: float  # Main result value (e.g., probability, edge)
    confidence: float  # Confidence in the result (0-1)
    
    # Signal information
    signal: Optional[str] = None  # e.g., 'BUY_YES', 'BUY_NO', 'HOLD'
    signal_strength: SignalStrength = SignalStrength.NEUTRAL
    
    # Supporting data
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if the result is valid."""
        return len(self.errors) == 0 and self.confidence > 0
    
    @property
    def is_actionable(self) -> bool:
        """Check if the result suggests action."""
        return (
            self.is_valid and 
            self.signal is not None and 
            self.signal != 'HOLD' and
            self.signal_strength.value >= SignalStrength.MODERATE.value
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'analysis_type': self.analysis_type.value,
            'timestamp': self.timestamp.isoformat(),
            'market_id': self.market_id,
            'value': self.value,
            'confidence': self.confidence,
            'signal': self.signal,
            'signal_strength': self.signal_strength.value,
            'components': self.components,
            'metadata': self.metadata,
            'errors': self.errors,
            'warnings': self.warnings
        }


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analysis components.
    
    This class defines the interface that all analyzers must implement,
    ensuring consistency across different analysis types.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()
        self.config.validate()
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the analyzer.
        
        This method should perform any setup required before analysis,
        such as loading models or establishing data connections.
        """
        pass
    
    @abstractmethod
    async def analyze(
        self, 
        market_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Perform analysis on a market.
        
        Args:
            market_id: Market identifier
            data: Optional pre-loaded data
            
        Returns:
            AnalysisResult with findings
        """
        pass
    
    @abstractmethod
    async def batch_analyze(
        self,
        market_ids: List[str]
    ) -> List[AnalysisResult]:
        """
        Perform analysis on multiple markets.
        
        Args:
            market_ids: List of market identifiers
            
        Returns:
            List of AnalysisResults
        """
        pass
    
    async def validate_data(
        self,
        data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate input data for analysis.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for required fields (override in subclasses)
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check data quality
        if 'prices' in data and len(data['prices']) < self.config.min_data_points:
            errors.append(f"Insufficient data points: {len(data['prices'])} < {self.config.min_data_points}")
        
        return len(errors) == 0, errors
    
    def get_required_fields(self) -> List[str]:
        """
        Get list of required data fields.
        
        Override in subclasses to specify requirements.
        
        Returns:
            List of field names
        """
        return []
    
    async def cleanup(self) -> None:
        """
        Cleanup resources.
        
        This method should release any resources held by the analyzer.
        """
        self._initialized = False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(config={self.config})"


class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    This class defines the interface for components that provide
    data to analyzers.
    """
    
    @abstractmethod
    async def get_market_data(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get market data for analysis.
        
        Args:
            market_id: Market identifier
            start_time: Start of time window
            end_time: End of time window
            
        Returns:
            Dictionary with market data
        """
        pass
    
    @abstractmethod
    async def get_external_data(
        self,
        data_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get external data (e.g., sportsbook odds, weather).
        
        Args:
            data_type: Type of external data
            params: Parameters for data retrieval
            
        Returns:
            Dictionary with external data
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if data source is available.
        
        Returns:
            True if available, False otherwise
        """
        pass


class AnalysisPipeline:
    """
    Orchestrates multiple analyzers in a pipeline.
    
    This class allows chaining multiple analysis components
    to create complex analysis workflows.
    """
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.analyzers: List[BaseAnalyzer] = []
        self.results: List[AnalysisResult] = []
        
    def add_analyzer(self, analyzer: BaseAnalyzer) -> 'AnalysisPipeline':
        """
        Add an analyzer to the pipeline.
        
        Args:
            analyzer: Analyzer to add
            
        Returns:
            Self for chaining
        """
        self.analyzers.append(analyzer)
        return self
    
    async def run(
        self,
        market_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> List[AnalysisResult]:
        """
        Run all analyzers in the pipeline.
        
        Args:
            market_id: Market to analyze
            data: Optional pre-loaded data
            
        Returns:
            List of results from all analyzers
        """
        self.results = []
        
        for analyzer in self.analyzers:
            try:
                # Initialize if needed
                if not analyzer._initialized:
                    await analyzer.initialize()
                    analyzer._initialized = True
                
                # Run analysis
                result = await analyzer.analyze(market_id, data)
                self.results.append(result)
                
                # Pass results to next analyzer via data
                if data is None:
                    data = {}
                data[f'{analyzer.__class__.__name__}_result'] = result
                
            except Exception as e:
                logger.error(f"Error in {analyzer.__class__.__name__}: {e}")
                # Create error result
                error_result = AnalysisResult(
                    analysis_type=AnalysisType.EDGE_DETECTION,
                    timestamp=datetime.now(),
                    market_id=market_id,
                    value=0,
                    confidence=0,
                    errors=[str(e)]
                )
                self.results.append(error_result)
        
        return self.results
    
    def get_combined_signal(self) -> Tuple[Optional[str], float]:
        """
        Combine signals from all analyzers.
        
        Returns:
            Tuple of (combined_signal, confidence)
        """
        if not self.results:
            return None, 0
        
        # Count signals
        signal_counts = {}
        total_confidence = 0
        
        for result in self.results:
            if result.is_actionable:
                signal = result.signal
                confidence = result.confidence * result.signal_strength.value / 5
                
                if signal not in signal_counts:
                    signal_counts[signal] = 0
                signal_counts[signal] += confidence
                total_confidence += confidence
        
        if not signal_counts:
            return 'HOLD', 0
        
        # Get strongest signal
        best_signal = max(signal_counts, key=signal_counts.get)
        combined_confidence = signal_counts[best_signal] / len(self.results)
        
        return best_signal, min(combined_confidence, 1.0)