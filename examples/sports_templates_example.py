#!/usr/bin/env python3
"""
Example usage of Sports Prediction Strategy Templates

This script demonstrates how to use the pre-built sports prediction templates
for NFL and college football trading strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_sdk.templates.sports.sentiment_momentum import SentimentMomentumTemplate
from neural_sdk.templates.sports.weather_impact import WeatherImpactTemplate

def example_sentiment_momentum():
    """Example using Sentiment Momentum Template."""
    print("🏈 Sentiment Momentum Template Example")
    print("=" * 50)

    # Create template with custom parameters
    template = SentimentMomentumTemplate(
        sentiment_threshold=0.7,
        momentum_period=15,
        sentiment_weight=0.6,
        sport='nfl'
    )

    print(f"Template: {template.name}")
    print(f"Sport: {template.sport}")
    print(f"Description: {template.description}")
    print(f"Parameters: {list(template.parameters.keys())}")

    # Example market data (simulated)
    market_data = {
        'markets': [
            {
                'ticker': 'KXNFLCHIEFS',
                'last_price': 0.65,
                'volatility': 0.1
            },
            {
                'ticker': 'KXNFLCHARGERS',
                'last_price': 0.55,
                'volatility': 0.15
            }
        ],
        'price_history': {
            'KXNFLCHIEFS': [0.60, 0.62, 0.64, 0.63, 0.65],
            'KXNFLCHARGERS': [0.50, 0.52, 0.54, 0.53, 0.55]
        }
    }

    # Generate signals
    signals = template.generate_signals(market_data)

    print(f"\nGenerated {len(signals)} trading signals:")
    for signal in signals:
        print(f"  📈 {signal['action']} {signal['symbol']} - Size: {signal['size']:.3f}, Confidence: {signal['confidence']:.2f}")
        print(f"     Reason: {signal['reason']}")

    print()

def example_weather_impact():
    """Example using Weather Impact Template."""
    print("🌤️ Weather Impact Template Example")
    print("=" * 50)

    # Create template
    template = WeatherImpactTemplate(
        temperature_weight=0.3,
        wind_weight=0.4,
        precipitation_weight=0.3,
        sport='nfl'
    )

    print(f"Template: {template.name}")
    print(f"Sport: {template.sport}")
    print(f"Description: {template.description}")

    # Example market data
    market_data = {
        'markets': [
            {
                'ticker': 'KXNFLCHIEFS',
                'last_price': 0.60,
                'volatility': 0.12
            }
        ]
    }

    # Generate signals
    signals = template.generate_signals(market_data)

    print(f"\nGenerated {len(signals)} trading signals:")
    for signal in signals:
        print(f"  🌧️ {signal['action']} {signal['symbol']} - Size: {signal['size']:.3f}, Confidence: {signal['confidence']:.2f}")
        print(f"     Reason: {signal['reason']}")

    print()

def example_parameter_customization():
    """Example of parameter validation and customization."""
    print("⚙️ Parameter Customization Example")
    print("=" * 50)

    template = SentimentMomentumTemplate()

    # Valid parameters
    valid_params = {
        'sentiment_threshold': 0.8,
        'momentum_period': 20,
        'sentiment_weight': 0.7
    }

    print("Testing parameter validation:")
    print(f"Valid parameters: {template.validate_parameters(valid_params)}")

    # Invalid parameters
    invalid_params = {
        'sentiment_threshold': 1.5,  # > 1.0
        'momentum_period': 5,        # < 5
    }

    print(f"Invalid parameters: {template.validate_parameters(invalid_params)}")

    # Show parameter definitions
    print("\nParameter definitions:")
    for param_name, param_info in template.parameters.items():
        print(f"  {param_name}: {param_info['description']}")
        print(f"    Range: {param_info['min']} - {param_info['max']}")
        print(f"    Default: {param_info['default']}")

    print()

def main():
    """Run all examples."""
    print("🎯 Neural SDK Sports Prediction Templates")
    print("=" * 60)
    print("Examples demonstrating NFL trading strategy templates")
    print("with sentiment analysis and weather impact factors.\n")

    try:
        example_sentiment_momentum()
        example_weather_impact()
        example_parameter_customization()

        print("✅ All examples completed successfully!")
        print("\n💡 Next steps:")
        print("  1. Customize template parameters for your strategy")
        print("  2. Integrate with real market data feeds")
        print("  3. Add sentiment data sources (Twitter, news)")
        print("  4. Backtest with historical data")
        print("  5. Deploy to live trading")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()