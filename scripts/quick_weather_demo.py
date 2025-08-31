#!/usr/bin/env python3
"""
Quick demo of weather monitoring for game venues
"""

import asyncio
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sdk.adapters.weather import WeatherAdapter, WeatherImpactAnalyzer


async def main():
    """Quick weather demo"""
    
    print("="*60)
    print("🌤️  WEATHER MONITORING DEMO - GAME VENUES")
    print("="*60)
    
    # NFL stadiums to monitor
    stadiums = [
        {"name": "Arrowhead Stadium", "team": "KC Chiefs", "lat": 39.0489, "lon": -94.4839, "outdoor": True},
        {"name": "Highmark Stadium", "team": "Buffalo Bills", "lat": 42.7738, "lon": -78.7870, "outdoor": True},
        {"name": "Lambeau Field", "team": "Green Bay", "lat": 44.5013, "lon": -88.0622, "outdoor": True},
        {"name": "Mile High", "team": "Denver", "lat": 39.7439, "lon": -105.0201, "outdoor": True}
    ]
    
    config = {
        'api_key': '78596505b0f5fea89e98ebcbf3bd6e21',
        'stadiums': stadiums
    }
    
    adapter = WeatherAdapter(config)
    analyzer = WeatherImpactAnalyzer()
    
    try:
        # Connect
        connected = await adapter.connect()
        if not connected:
            print("❌ Failed to connect")
            return
            
        print("✅ Connected to OpenWeatherMap\n")
        
        # Check each stadium
        for stadium in stadiums:
            print(f"\n📍 {stadium['name']} ({stadium['team']})")
            print("-" * 40)
            
            # Get weather
            conditions = await adapter._fetch_weather(stadium)
            
            if conditions:
                # Display conditions
                print(f"🌡️  Temperature: {conditions['temperature']:.0f}°F (feels like {conditions['feels_like']:.0f}°F)")
                print(f"💨 Wind: {conditions['wind_speed']:.0f} mph")
                print(f"💧 Humidity: {conditions['humidity']}%")
                print(f"🌧️  Precipitation: {conditions['precipitation']:.2f} in/hr")
                print(f"👁️  Visibility: {conditions['visibility']:.1f} miles")
                print(f"☁️  Conditions: {conditions['weather']} - {conditions['description']}")
                
                # Analyze impact for NFL
                impact = analyzer.calculate_total_impact(conditions, "NFL")
                
                print(f"\n⚡ Game Impact Analysis:")
                print(f"   Overall: {impact['overall_impact'].upper()}")
                print(f"   Score: {impact['impact_score']:.1f}/10")
                print(f"   💰 Betting: {impact['betting_recommendation']}")
                
                # Check for significant conditions
                if conditions['wind_speed'] > 15:
                    print(f"   ⚠️  HIGH WIND: Affects passing game & field goals")
                if conditions['precipitation'] > 0.1:
                    print(f"   ⚠️  PRECIPITATION: Affects ball handling & footing")
                if conditions['visibility'] < 5:
                    print(f"   ⚠️  LOW VISIBILITY: Affects deep passes")
                    
            await asyncio.sleep(0.5)  # Small delay between API calls
            
        # Disconnect
        await adapter.disconnect()
        print("\n" + "="*60)
        print("✅ Demo complete! Weather monitoring is working.")
        print("\nThis data updates every 5 minutes during games to detect:")
        print("• Wind changes that affect passing/kicking")
        print("• Precipitation that impacts scoring")
        print("• Temperature swings affecting player performance")
        print("• Visibility issues from fog/snow")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        await adapter.disconnect()


if __name__ == "__main__":
    print("""
    This demo shows real-time weather monitoring at NFL stadiums.
    The system detects conditions that impact game outcomes and
    generates trading signals when weather creates betting edges.
    """)
    
    asyncio.run(main())