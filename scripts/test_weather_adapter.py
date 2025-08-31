#!/usr/bin/env python3
"""
Test Weather Adapter with OpenWeatherMap API
Tests the weather adapter implementation with real API calls
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdk.adapters.weather import WeatherAdapter, WeatherImpactAnalyzer
from src.sdk.core.base_adapter import EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_weather_event(event):
    """Pretty print a weather event"""
    print("\n" + "="*60)
    print(f"🌤️  WEATHER EVENT DETECTED")
    print("="*60)
    
    data = event.data
    metadata = event.metadata if hasattr(event, 'metadata') else {}
    
    print(f"📍 Stadium: {data.get('stadium', 'Unknown')}")
    print(f"🏈 Team: {data.get('team', 'Unknown')}")
    print(f"⚠️  Condition: {data.get('condition', 'Unknown')}")
    print(f"📊 Impact Level: {event.impact}")
    print(f"🎯 Confidence: {event.confidence:.1%}")
    
    # Condition-specific details
    if data.get('condition') == 'high_wind':
        print(f"\n💨 Wind Details:")
        print(f"   Speed: {data.get('wind_speed', 0):.1f} mph")
        print(f"   Gusts: {data.get('wind_gust', 0):.1f} mph")
        print(f"   Direction: {data.get('wind_direction', 0)}°")
        
        impact = data.get('impact', {})
        print(f"\n📉 Game Impact:")
        print(f"   Passing: -{impact.get('passing_yards_reduction', 'N/A')}")
        print(f"   Field Goals: -{impact.get('field_goal_accuracy_reduction', 'N/A')}")
        print(f"   Punts: {impact.get('punt_distance_variance', 'N/A')}")
        print(f"   Strategy: {impact.get('recommended_strategy', 'N/A')}")
        
    elif data.get('condition') == 'precipitation':
        print(f"\n🌧️ Precipitation Details:")
        print(f"   Rate: {data.get('rate', 0):.2f} inches/hour")
        print(f"   Type: {data.get('type', 'Unknown')}")
        print(f"   Description: {data.get('description', 'N/A')}")
        
    elif data.get('condition') == 'temperature_change':
        print(f"\n🌡️ Temperature Change:")
        print(f"   Current: {data.get('current_temp', 0):.1f}°F")
        print(f"   Previous: {data.get('previous_temp', 0):.1f}°F")
        print(f"   Change: {data.get('change', 0):.1f}°F")
        
    elif data.get('condition') == 'low_visibility':
        print(f"\n🌫️ Visibility:")
        print(f"   Distance: {data.get('visibility', 0):.1f} miles")
        print(f"   Weather: {data.get('weather', 'Unknown')}")
    
    if metadata.get('affects'):
        print(f"\n🎮 Affects: {', '.join(metadata['affects'])}")
    if metadata.get('advantage'):
        print(f"✅ Advantage: {metadata['advantage']}")
    
    print("="*60)


async def test_basic_connection():
    """Test basic API connection"""
    print("\n" + "="*60)
    print("TEST 1: Basic API Connection")
    print("="*60)
    
    # Get API key from environment
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENWEATHER_API_KEY environment variable not set")
        print("Please set: export OPENWEATHER_API_KEY=your_api_key_here")
        return False
    
    config = {
        'api_key': api_key,
        'update_interval': 5  # Quick updates for testing
    }
    
    adapter = WeatherAdapter(config)
    
    try:
        # Test connection
        connected = await adapter.connect()
        if connected:
            print("✅ Successfully connected to OpenWeatherMap API")
            
            # Validate connection
            valid = await adapter.validate_connection()
            print(f"✅ Connection validation: {'Passed' if valid else 'Failed'}")
            
            # Check metadata
            metadata = adapter.get_metadata()
            print(f"\n📋 Adapter Metadata:")
            print(f"   Name: {metadata.name}")
            print(f"   Version: {metadata.version}")
            print(f"   Type: {metadata.source_type}")
            print(f"   Latency: {metadata.latency_ms}ms")
            print(f"   Reliability: {metadata.reliability:.1%}")
            
            await adapter.disconnect()
            return True
        else:
            print("❌ Failed to connect to OpenWeatherMap API")
            return False
            
    except Exception as e:
        print(f"❌ Error during connection test: {e}")
        return False


async def test_single_stadium():
    """Test fetching weather for a single stadium"""
    print("\n" + "="*60)
    print("TEST 2: Single Stadium Weather Check")
    print("="*60)
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return False
    
    # Test with Arrowhead Stadium (Kansas City)
    config = {
        'api_key': api_key,
        'stadiums': [
            {"name": "Arrowhead Stadium", "team": "KC", "lat": 39.0489, "lon": -94.4839, "outdoor": True}
        ]
    }
    
    adapter = WeatherAdapter(config)
    
    try:
        await adapter.connect()
        
        # Fetch weather directly
        stadium = config['stadiums'][0]
        conditions = await adapter._fetch_weather(stadium)
        
        if conditions:
            print(f"✅ Weather data received for {stadium['name']}")
            print(f"\n🌡️ Current Conditions:")
            print(f"   Temperature: {conditions['temperature']:.1f}°F")
            print(f"   Feels Like: {conditions['feels_like']:.1f}°F")
            print(f"   Humidity: {conditions['humidity']}%")
            print(f"   Wind Speed: {conditions['wind_speed']:.1f} mph")
            print(f"   Wind Direction: {conditions['wind_direction']}°")
            print(f"   Precipitation: {conditions['precipitation']:.2f} in/hr")
            print(f"   Visibility: {conditions['visibility']:.1f} miles")
            print(f"   Weather: {conditions['weather']}")
            print(f"   Description: {conditions['description']}")
            
            # Test impact analyzer
            analyzer = WeatherImpactAnalyzer()
            impact = analyzer.calculate_total_impact(conditions, "NFL")
            
            print(f"\n🎯 Impact Analysis:")
            print(f"   Overall Impact: {impact['overall_impact']}")
            print(f"   Impact Score: {impact['impact_score']:.1f}")
            print(f"   Factors: {', '.join(impact['factors']) if impact['factors'] else 'None'}")
            print(f"   Recommendation: {impact['betting_recommendation']}")
            
            await adapter.disconnect()
            return True
        else:
            print("❌ Failed to fetch weather data")
            await adapter.disconnect()
            return False
            
    except Exception as e:
        print(f"❌ Error during single stadium test: {e}")
        return False


async def test_event_generation():
    """Test event generation with different weather conditions"""
    print("\n" + "="*60)
    print("TEST 3: Event Generation (30 seconds)")
    print("="*60)
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return False
    
    # Multiple stadiums for better chance of events
    config = {
        'api_key': api_key,
        'update_interval': 10,  # Check every 10 seconds
        'stadiums': [
            {"name": "Arrowhead Stadium", "team": "KC", "lat": 39.0489, "lon": -94.4839, "outdoor": True},
            {"name": "Highmark Stadium", "team": "BUF", "lat": 42.7738, "lon": -78.7870, "outdoor": True},
            {"name": "Lambeau Field", "team": "GB", "lat": 44.5013, "lon": -88.0622, "outdoor": True},
            {"name": "Mile High", "team": "DEN", "lat": 39.7439, "lon": -105.0201, "outdoor": True}
        ],
        'thresholds': {
            'wind_speed': 10,  # Lower threshold for testing
            'precipitation': 0.01,  # Lower threshold for testing
            'temperature_change': 5,  # Lower threshold for testing
            'visibility': 5  # Lower threshold for testing
        }
    }
    
    adapter = WeatherAdapter(config)
    
    try:
        await adapter.connect()
        await adapter.start()  # Start the adapter
        
        print("🔍 Monitoring weather conditions...")
        print("   (Lower thresholds set for demonstration)")
        print("   Checking 4 stadiums every 10 seconds...")
        print("-" * 60)
        
        event_count = 0
        start_time = asyncio.get_event_loop().time()
        duration = 30  # seconds
        
        async for event in adapter.stream():
            if event.event_type == EventType.WEATHER_UPDATE:
                event_count += 1
                print_weather_event(event)
            
            # Check if duration exceeded
            if asyncio.get_event_loop().time() - start_time > duration:
                break
        
        print(f"\n📊 Summary:")
        print(f"   Events Generated: {event_count}")
        print(f"   Duration: {duration} seconds")
        
        stats = adapter.statistics
        print(f"   Total API Calls: {stats.get('event_count', 0) + stats.get('error_count', 0)}")
        print(f"   Errors: {stats.get('error_count', 0)}")
        
        await adapter.stop()
        return True
        
    except Exception as e:
        print(f"❌ Error during event generation test: {e}")
        return False


async def test_extreme_weather():
    """Simulate extreme weather conditions"""
    print("\n" + "="*60)
    print("TEST 4: Extreme Weather Simulation")
    print("="*60)
    
    # Create analyzer
    analyzer = WeatherImpactAnalyzer()
    
    # Test different extreme conditions
    scenarios = [
        {
            "name": "Blizzard Conditions",
            "conditions": {
                "wind_speed": 35,
                "precipitation": 0.5,
                "temperature": 15,
                "visibility": 0.25
            },
            "sport": "NFL"
        },
        {
            "name": "Heavy Rain",
            "conditions": {
                "wind_speed": 10,
                "precipitation": 0.3,
                "temperature": 55,
                "visibility": 2
            },
            "sport": "NFL"
        },
        {
            "name": "Extreme Heat",
            "conditions": {
                "wind_speed": 5,
                "precipitation": 0,
                "temperature": 105,
                "humidity": 85
            },
            "sport": "MLB"
        },
        {
            "name": "High Winds",
            "conditions": {
                "wind_speed": 30,
                "precipitation": 0,
                "temperature": 65,
                "visibility": 10
            },
            "sport": "NFL"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🌪️ Scenario: {scenario['name']}")
        print(f"   Sport: {scenario['sport']}")
        print(f"   Conditions:")
        for key, value in scenario['conditions'].items():
            unit = "mph" if "wind" in key else "in/hr" if "precip" in key else "°F" if "temp" in key else "miles" if "vis" in key else "%"
            print(f"     {key}: {value} {unit}")
        
        impact = analyzer.calculate_total_impact(scenario['conditions'], scenario['sport'])
        
        print(f"   📊 Impact Assessment:")
        print(f"     Overall: {impact['overall_impact']}")
        print(f"     Score: {impact['impact_score']:.1f}")
        print(f"     Recommendation: {impact['betting_recommendation']}")
    
    return True


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🌤️  WEATHER ADAPTER TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for API key
    if not os.getenv('OPENWEATHER_API_KEY'):
        print("\n⚠️  WARNING: No API key found!")
        print("To test with real API:")
        print("1. Sign up at https://openweathermap.org/api")
        print("2. Get your free API key")
        print("3. Set environment variable:")
        print("   export OPENWEATHER_API_KEY=your_key_here")
        print("\nRunning simulation tests only...")
        
        # Run only simulation tests
        await test_extreme_weather()
        return
    
    # Run all tests
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Single Stadium", test_single_stadium),
        ("Event Generation", test_event_generation),
        ("Extreme Weather", test_extreme_weather)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            result = await test_func()
            results[name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            results[name] = f"❌ ERROR: {e}"
            logger.error(f"Test {name} failed with error: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║         WEATHER ADAPTER TEST - NEURAL TRADING        ║
    ║                                                       ║
    ║  This script tests the OpenWeatherMap integration    ║
    ║  for monitoring weather conditions at game venues.   ║
    ║                                                       ║
    ║  Tests include:                                      ║
    ║  • API connection and authentication                 ║
    ║  • Weather data fetching                            ║
    ║  • Event generation based on thresholds             ║
    ║  • Impact analysis for trading decisions            ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())