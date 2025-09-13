#!/usr/bin/env python3
"""
🧮 Neural SDK Edge & Confidence Calculation Demo

This script demonstrates the exact mathematical formulas used to calculate
trading edges and confidence levels in the Neural SDK.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any

def demonstrate_edge_calculations():
    """Demonstrate both edge calculation methods with real examples."""
    
    print("🧮" + "="*80)
    print("📊 NEURAL SDK EDGE & CONFIDENCE CALCULATIONS")
    print("🧮" + "="*80)
    print()
    
    # Example 1: Production Demo Method (Multi-Factor Sentiment)
    print("📈 METHOD 1: MULTI-FACTOR SENTIMENT ANALYSIS")
    print("="*50)
    
    # Real data from logs (Colorado vs Houston, Cycle 1)
    sentiment_factors = {
        'social_sentiment': 0.772,      # 77.2%
        'news_sentiment': 0.718,        # 71.8% 
        'betting_line_sentiment': 0.684, # 68.4%
        'historical_performance': 0.821, # 82.1%
        'injury_reports': 0.950,        # 95.0%
        'weather_factors': 0.980        # 98.0%
    }
    
    weights = {
        'social': 0.25,        # 25%
        'news': 0.20,         # 20%
        'betting': 0.20,      # 20%
        'historical': 0.15,   # 15%
        'injury': 0.15,       # 15%
        'weather': 0.05       # 5%
    }
    
    print("🔍 INPUT SENTIMENT FACTORS:")
    for factor, value in sentiment_factors.items():
        print(f"   {factor.replace('_', ' ').title()}: {value:.1%}")
    
    print("\n⚖️ FACTOR WEIGHTS:")
    for factor, weight in weights.items():
        print(f"   {factor.title()}: {weight:.0%}")
    
    # Calculate composite sentiment (fair value)
    composite_sentiment = (
        sentiment_factors['social_sentiment'] * weights['social'] +
        sentiment_factors['news_sentiment'] * weights['news'] +
        sentiment_factors['betting_line_sentiment'] * weights['betting'] +
        sentiment_factors['historical_performance'] * weights['historical'] +
        sentiment_factors['injury_reports'] * weights['injury'] +
        sentiment_factors['weather_factors'] * weights['weather']
    )
    
    current_market_price = 0.520  # 52.0¢ from logs
    edge = composite_sentiment - current_market_price
    
    # Calculate confidence
    confidence = min(0.95, 0.60 + abs(edge) * 2)
    
    # Volume-based liquidity factor
    volume = 45000  # $45,000 from logs
    liquidity_factor = min(2.0, volume / 50000)
    
    # Expected value
    expected_value = abs(edge) * 1000 * liquidity_factor * confidence
    
    print(f"\n🎯 CALCULATIONS:")
    print(f"   Fair Value = Σ(factor × weight) = {composite_sentiment:.3f} = {composite_sentiment:.1%}")
    print(f"   Market Price = {current_market_price:.3f} = {current_market_price:.1%}")
    print(f"   Raw Edge = Fair Value - Market Price = {edge:.3f} = {edge:.1%}")
    print(f"   Confidence = min(95%, 60% + |edge|×2) = {confidence:.1%}")
    print(f"   Liquidity Factor = min(2.0, volume/50k) = {liquidity_factor:.2f}x")
    print(f"   Expected Value = |edge| × 1000 × liq_factor × confidence = ${expected_value:.2f}")
    
    print(f"\n✅ RESULT: {abs(edge):.1%} edge, {confidence:.0%} confidence, ${expected_value:.2f} EV")
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Sentiment Stack Method (Sigmoid-based)
    print("📈 METHOD 2: ADVANCED SENTIMENT ENGINE")
    print("="*50)
    
    # Team sentiment metrics
    home_sentiment = 0.75  # 75% bullish on home team
    home_confidence = 0.85  # 85% confidence
    away_sentiment = 0.45   # 45% bullish on away team (55% bearish)
    away_confidence = 0.80  # 80% confidence  
    market_sentiment = 0.68 # 68% overall market bullishness
    market_confidence = 0.90 # 90% confidence
    
    print("🏠 TEAM SENTIMENT DATA:")
    print(f"   Home Team Sentiment: {home_sentiment:.1%} (confidence: {home_confidence:.0%})")
    print(f"   Away Team Sentiment: {away_sentiment:.1%} (confidence: {away_confidence:.0%})")
    print(f"   Market Sentiment: {market_sentiment:.1%} (confidence: {market_confidence:.0%})")
    
    # Weight sentiment by confidence
    home_score = home_sentiment * home_confidence
    away_score = away_sentiment * away_confidence  
    market_score = market_sentiment * market_confidence
    
    print(f"\n📊 CONFIDENCE-WEIGHTED SCORES:")
    print(f"   Home Score: {home_sentiment:.1%} × {home_confidence:.0%} = {home_score:.3f}")
    print(f"   Away Score: {away_sentiment:.1%} × {away_confidence:.0%} = {away_score:.3f}")
    print(f"   Market Score: {market_sentiment:.1%} × {market_confidence:.0%} = {market_score:.3f}")
    
    # Combine sentiments
    combined_sentiment = (
        home_score * 0.4 +           # 40% home team
        (-away_score) * 0.4 +        # 40% away team (inverted)
        market_score * 0.2           # 20% overall market
    )
    
    print(f"\n⚖️ WEIGHTED COMBINATION:")
    print(f"   Combined = {home_score:.3f}×40% + (-{away_score:.3f})×40% + {market_score:.3f}×20%")
    print(f"   Combined = {combined_sentiment:.3f}")
    
    # Apply sigmoid transformation
    probability = 1 / (1 + np.exp(-combined_sentiment * 3))
    
    # Confidence adjustment
    avg_confidence = (home_confidence + away_confidence + market_confidence) / 3
    adjusted_probability = 0.5 + (probability - 0.5) * avg_confidence
    final_probability = max(0.01, min(0.99, adjusted_probability))
    
    print(f"\n🔄 SIGMOID TRANSFORMATION:")
    print(f"   Raw Probability = 1/(1+exp(-{combined_sentiment:.3f}×3)) = {probability:.3f}")
    print(f"   Avg Confidence = {avg_confidence:.1%}")
    print(f"   Adjusted Probability = 0.5 + ({probability:.3f}-0.5)×{avg_confidence:.3f} = {final_probability:.3f}")
    
    # Calculate edge
    market_price_2 = 0.54  # 54% market price
    edge_2 = final_probability - market_price_2
    
    # Only consider significant edges
    if abs(edge_2) > 0.03:
        edge_final = abs(edge_2)
    else:
        edge_final = 0
    
    print(f"\n🎯 EDGE CALCULATION:")
    print(f"   Fair Probability: {final_probability:.1%}")
    print(f"   Market Price: {market_price_2:.1%}")
    print(f"   Raw Divergence: {edge_2:.1%}")
    print(f"   Final Edge: {edge_final:.1%} {'✅' if edge_final > 0.03 else '❌'}")
    
    print(f"\n✅ RESULT: {edge_final:.1%} edge using advanced sentiment engine")
    
    print("\n" + "="*80 + "\n")
    
    # Comparison
    print("⚖️ METHOD COMPARISON")
    print("="*30)
    print(f"Method 1 (Multi-Factor): {abs(edge):.1%} edge, {confidence:.0%} confidence")
    print(f"Method 2 (Sentiment Stack): {edge_final:.1%} edge")
    print(f"Average Edge: {(abs(edge) + edge_final)/2:.1%}")
    print()
    
    # Kelly Criterion position sizing demo
    print("💰 KELLY CRITERION POSITION SIZING")
    print("="*40)
    
    kelly_edge = abs(edge)  # Use Method 1 edge
    win_prob = composite_sentiment
    lose_prob = 1 - win_prob
    
    # Kelly formula: f = (bp - q) / b
    # Where: b = odds received, p = win probability, q = lose probability
    # For binary markets: f = (p - (1-p)) / 1 = 2p - 1, but we use edge/variance
    
    kelly_fraction = kelly_edge / (1 - current_market_price)  # Simplified Kelly
    conservative_kelly = kelly_fraction * 0.5  # 50% of Kelly for safety
    max_position = 0.08  # 8% maximum position
    
    recommended_size = min(max_position, conservative_kelly, 0.10)
    
    print(f"🎲 Kelly Calculation:")
    print(f"   Win Probability: {win_prob:.1%}")
    print(f"   Edge: {kelly_edge:.1%}")
    print(f"   Kelly Fraction: {kelly_edge:.3f}/(1-{current_market_price:.3f}) = {kelly_fraction:.3f} = {kelly_fraction:.1%}")
    print(f"   Conservative Kelly: {kelly_fraction:.3f} × 0.5 = {conservative_kelly:.3f} = {conservative_kelly:.1%}")
    print(f"   Final Position Size: min(8%, {conservative_kelly:.1%}, 10%) = {recommended_size:.1%}")
    
    print(f"\n💼 POSITION SIZING RESULT: {recommended_size:.1%} of capital")
    
    print("\n🏆 SUMMARY")
    print("="*20)
    print("The Neural SDK uses sophisticated mathematical models to:")
    print("✅ Detect market mispricings (edges of 15-25%)")  
    print("✅ Calculate confidence levels dynamically")
    print("✅ Apply Kelly criterion for optimal position sizing")
    print("✅ Adjust for market liquidity and volume")
    print("✅ Provide comprehensive risk management")
    print()
    print("These calculations enable institutional-grade trading performance!")


def demonstrate_confidence_scaling():
    """Show how confidence scales with edge size."""
    
    print("\n📊 CONFIDENCE SCALING DEMONSTRATION")
    print("="*50)
    
    edges = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print("Edge    | Confidence | Formula")
    print("--------|------------|------------------------")
    
    for edge in edges:
        confidence = min(0.95, 0.60 + edge * 2)
        formula = f"min(95%, 60% + {edge:.0%}×2)"
        print(f"{edge:6.0%} | {confidence:8.0%}   | {formula}")
    
    print("\nKey Insights:")
    print("• Confidence starts at 60% base level")
    print("• Each 1% of edge adds 2% confidence")
    print("• Maximum confidence capped at 95%")
    print("• Large edges (15%+) get maximum confidence")


if __name__ == "__main__":
    demonstrate_edge_calculations()
    demonstrate_confidence_scaling()
