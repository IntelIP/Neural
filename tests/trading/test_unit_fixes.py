"""
Quick fixes for unit test failures

This script will patch the common issues found in our test suite.
"""

import sys
import os

# Add project root to path  
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from datetime import datetime, timezone
from neural.strategy.base import Signal, SignalType

def create_test_signal(
    signal_type: SignalType = SignalType.BUY_YES,
    market_id: str = "TEST-MARKET",
    confidence: float = 0.75,
    edge: float = 0.05,
    recommended_size: float = 0.08,
    expected_value: float = 50.0,
    max_contracts: int = 1000
) -> Signal:
    """Helper function to create test signals with all required parameters."""
    return Signal(
        signal_type=signal_type,
        market_id=market_id,
        timestamp=datetime.now(timezone.utc),
        confidence=confidence,
        edge=edge,
        expected_value=expected_value,
        recommended_size=recommended_size,
        max_contracts=max_contracts
    )

if __name__ == "__main__":
    # Test signal creation
    signal = create_test_signal()
    print(f"✅ Test signal created: {signal.signal_type.value}")
    print(f"   Market: {signal.market_id}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Edge: {signal.edge:.1%}")
    print(f"   Size: {signal.recommended_size:.1%}")
    print(f"   Expected Value: ${signal.expected_value}")
    print(f"   Max Contracts: {signal.max_contracts}")
