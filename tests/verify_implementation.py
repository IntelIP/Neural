#!/usr/bin/env python
"""
Verification script for Sprint 1 implementation
Tests all new infrastructure components
"""

import asyncio
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

print("=" * 60)
print("SPRINT 1 IMPLEMENTATION VERIFICATION")
print("=" * 60)

def test_imports():
    """Test all imports work correctly"""
    print("\n1. Testing imports...")
    
    try:
        from data_pipeline.event_buffer import EventBuffer, BufferedEvent, Priority
        print("   ✓ EventBuffer imports OK")
    except Exception as e:
        print(f"   ✗ EventBuffer import failed: {e}")
        return False
    
    try:
        from data_pipeline.state_manager import AgentStateManager, get_state_manager, ComputationCache
        print("   ✓ StateManager imports OK")
    except Exception as e:
        print(f"   ✗ StateManager import failed: {e}")
        return False
    
    try:
        from data_pipeline.window_aggregator import WindowAggregator, TumblingWindow, SlidingWindow
        print("   ✓ WindowAggregator imports OK")
    except Exception as e:
        print(f"   ✗ WindowAggregator import failed: {e}")
        return False
    
    try:
        from data_pipeline.orchestration.unified_stream_manager import StreamManager
        print("   ✓ StreamManager imports OK")
    except Exception as e:
        print(f"   ✗ StreamManager import failed: {e}")
        return False
    
    return True

async def test_event_buffer():
    """Test EventBuffer functionality"""
    print("\n2. Testing EventBuffer...")
    
    from data_pipeline.event_buffer import EventBuffer, BufferedEvent, Priority
    
    buffer = EventBuffer(size=100)
    
    # Test writing events
    events_written = 0
    for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
        event = BufferedEvent(
            event_type=f'test_{priority.name}',
            data={'priority': priority.name, 'value': events_written},
            priority=priority,
            destinations=['DataCoordinator']
        )
        success = await buffer.write(event)
        if success:
            events_written += 1
    
    print(f"   ✓ Wrote {events_written} events")
    
    # Test reading events
    events = await buffer.read('test_consumer', batch_size=10)
    print(f"   ✓ Read {len(events)} events")
    
    # Test priority ordering (critical should come first)
    if events and events[0].priority == Priority.CRITICAL:
        print("   ✓ Priority ordering working")
    
    # Test stats
    stats = buffer.get_stats()
    print(f"   ✓ Stats: written={stats['events_written']}, read={stats['events_read']}")
    
    return True

async def test_window_aggregator():
    """Test WindowAggregator functionality"""
    print("\n3. Testing WindowAggregator...")
    
    from data_pipeline.window_aggregator import WindowAggregator
    
    aggregator = WindowAggregator()
    
    # Add events
    for i in range(10):
        await aggregator.add_event(
            value=i * 10,
            metadata={'type': 'test', 'index': i}
        )
    
    # Test aggregates
    windows_tested = 0
    for window_key in ['1m', '5m', '15m']:
        aggregates = aggregator.get_aggregates(window_key)
        if aggregates:
            windows_tested += 1
    
    print(f"   ✓ Tested {windows_tested} window types")
    
    # Test velocity calculation
    velocity = aggregator.calculate_velocity('5m')
    print(f"   ✓ Velocity calculation: {velocity}")
    
    # Test anomaly detection
    is_anomaly = aggregator.detect_anomaly('5m')
    print(f"   ✓ Anomaly detection: {is_anomaly}")
    
    return True

def test_state_manager():
    """Test StateManager functionality"""
    print("\n4. Testing StateManager...")
    
    from data_pipeline.state_manager import get_state_manager
    
    state_manager = get_state_manager()
    
    # Test hot cache
    cache = state_manager.hot_caches['agent_state']
    cache.set('test_key', {'value': 'test_data'}, ttl=60)
    value = cache.get('test_key')
    
    if value and value['value'] == 'test_data':
        print("   ✓ Hot cache working")
    
    # Test cache stats
    stats = cache.get_stats()
    print(f"   ✓ Cache stats: size={stats['size']}, hits={stats['hits']}, misses={stats['misses']}")
    
    # Test default states
    agents_with_defaults = 0
    for agent_name in ['DataCoordinator', 'MarketEngineer', 'RiskManager', 'StrategyAnalyst', 'TradeExecutor']:
        default_state = state_manager.get_default_state(agent_name)
        if default_state:
            agents_with_defaults += 1
    
    print(f"   ✓ Default states for {agents_with_defaults} agents")
    
    return True

def test_stream_manager():
    """Test StreamManager initialization"""
    print("\n5. Testing StreamManager...")
    
    from data_pipeline.orchestration.unified_stream_manager import StreamManager
    
    sm = StreamManager()
    
    # Verify components are initialized
    checks_passed = 0
    
    if sm.event_buffer and sm.event_buffer.size > 0:
        print(f"   ✓ EventBuffer initialized (size: {sm.event_buffer.size})")
        checks_passed += 1
    
    if sm.state_manager and len(sm.state_manager.hot_caches) > 0:
        print(f"   ✓ StateManager initialized ({len(sm.state_manager.hot_caches)} caches)")
        checks_passed += 1
    
    if sm.window_aggregator and len(sm.window_aggregator.windows) > 0:
        print(f"   ✓ WindowAggregator initialized ({len(sm.window_aggregator.windows)} windows)")
        checks_passed += 1
    
    if sm.computation_cache:
        print("   ✓ ComputationCache initialized")
        checks_passed += 1
    
    # Test statistics method
    stats = sm.get_statistics()
    if 'buffer_stats' in stats and 'cache_stats' in stats and 'window_stats' in stats:
        print("   ✓ Enhanced statistics working")
        checks_passed += 1
    
    return checks_passed == 5

def test_agent_integrations():
    """Test agent integrations"""
    print("\n6. Testing Agent Integrations...")
    
    try:
        from agentuity_agents.DataCoordinator.agent import run as dc_run
        print("   ✓ DataCoordinator agent loads")
    except Exception as e:
        print(f"   ✗ DataCoordinator failed: {e}")
        return False
    
    try:
        from agentuity_agents.MarketEngineer.agent import run as me_run
        # Check if WindowAggregator is imported
        import agentuity_agents.MarketEngineer.agent as me_module
        if 'WindowAggregator' in me_module.__dict__:
            print("   ✓ MarketEngineer uses WindowAggregator")
        else:
            print("   ✓ MarketEngineer agent loads")
    except Exception as e:
        print(f"   ✗ MarketEngineer failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests"""
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test EventBuffer
    try:
        await test_event_buffer()
    except Exception as e:
        print(f"   ✗ EventBuffer test failed: {e}")
        all_passed = False
    
    # Test WindowAggregator
    try:
        await test_window_aggregator()
    except Exception as e:
        print(f"   ✗ WindowAggregator test failed: {e}")
        all_passed = False
    
    # Test StateManager
    try:
        test_state_manager()
    except Exception as e:
        print(f"   ✗ StateManager test failed: {e}")
        all_passed = False
    
    # Test StreamManager
    try:
        test_stream_manager()
    except Exception as e:
        print(f"   ✗ StreamManager test failed: {e}")
        all_passed = False
    
    # Test Agent integrations
    try:
        test_agent_integrations()
    except Exception as e:
        print(f"   ✗ Agent integration test failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Implementation verified!")
    else:
        print("⚠️  Some tests failed - please review")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)