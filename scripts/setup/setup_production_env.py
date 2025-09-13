#!/usr/bin/env python3
"""
🚀 Neural SDK Production Environment Setup

This script helps you configure the production environment for live CFB trading.
"""

import os
import sys
from pathlib import Path

def create_env_template():
    """Create .env template file for production configuration."""
    env_template = """# =============================================================================
# NEURAL SDK PRODUCTION CONFIGURATION
# =============================================================================
# Copy this to .env and fill in your actual credentials

# Kalshi Production API Credentials
# Get these from: https://kalshi.com/profile/api
KALSHI_API_KEY=your_production_api_key_here
KALSHI_PRIVATE_KEY_PATH=/absolute/path/to/your/kalshi_private_key.pem

# Trading Configuration
TRADING_MODE=PAPER         # PAPER or LIVE (start with PAPER for testing!)
KALSHI_ENVIRONMENT=PRODUCTION
MAX_POSITION_SIZE=0.08     # 8% maximum position size
MIN_EDGE_THRESHOLD=0.04    # 4% minimum edge threshold  
MIN_CONFIDENCE_THRESHOLD=0.65  # 65% minimum confidence
MAX_ORDERS_PER_MINUTE=3    # Conservative rate limiting

# Portfolio Settings  
INITIAL_CAPITAL=10000      # Starting capital in USD
MAX_DAILY_LOSS=500         # Maximum daily loss limit
MAX_DRAWDOWN=0.15          # 15% maximum drawdown

# Optional: Twitter API for enhanced sentiment (leave blank to use synthetic)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Logging
LOG_LEVEL=INFO
"""
    
    env_file = Path('.env.example')
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.example template file")
    return env_file

def check_env_file():
    """Check if .env file exists and has required fields."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_fields = [
        'KALSHI_API_KEY',
        'KALSHI_PRIVATE_KEY_PATH',
        'TRADING_MODE'
    ]
    
    missing_fields = []
    for field in required_fields:
        if f'{field}=' not in content or f'{field}=your_' in content:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing or incomplete fields in .env: {missing_fields}")
        return False
    
    print("✅ .env file looks properly configured")
    return True

def install_dependencies():
    """Install required packages for production."""
    print("📦 Installing production dependencies...")
    
    packages = [
        'python-dotenv',  # For loading .env files
        'cryptography',   # For Kalshi authentication
        'aiohttp',        # For async HTTP requests
        'websockets',     # For WebSocket connections
    ]
    
    for package in packages:
        os.system(f'pip install {package}')
    
    print("✅ Dependencies installed")

def main():
    """Setup production environment."""
    print("🚀 Neural SDK Production Environment Setup")
    print("="*50)
    
    # Step 1: Create template
    template_file = create_env_template()
    print(f"📄 Template created: {template_file}")
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Check for actual .env
    print("\n📋 NEXT STEPS:")
    print("="*30)
    
    if not check_env_file():
        print("1️⃣ Copy .env.example to .env:")
        print("   cp .env.example .env")
        print()
        print("2️⃣ Edit .env file with your actual Kalshi credentials:")
        print("   • Get API key from: https://kalshi.com/profile/api")
        print("   • Download private key and set KALSHI_PRIVATE_KEY_PATH")
        print("   • Set TRADING_MODE=PAPER for initial testing")
        print()
        print("3️⃣ Run production trading:")
        print("   python examples/production_cfb_trading.py")
    else:
        print("✅ Environment configured! You can run:")
        print("   python examples/production_cfb_trading.py")
    
    print("\n⚠️  IMPORTANT SAFETY NOTES:")
    print("• Always start with TRADING_MODE=PAPER for testing")
    print("• Only use TRADING_MODE=LIVE when you're ready for real money")
    print("• Monitor your positions and set appropriate risk limits")
    print("• You can stop trading anytime with Ctrl+C")
    
    print(f"\n🚀 Production environment setup complete!")

if __name__ == "__main__":
    main()
