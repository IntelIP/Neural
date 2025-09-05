# 🚀 Neural SDK Team Installation Guide

**Quick setup for team members to start using Neural SDK v1.1.0 WebSocket streaming**

## 🎯 Quick Start (2 minutes)

### Option 1: One-Command Install
```bash
# Download and run installation script
curl -sSL https://raw.githubusercontent.com/IntelIP/Neural-Trading-Platform/main/scripts/install_neural_sdk.sh -o /tmp/install_neural.sh && \
bash /tmp/install_neural.sh
```

### Option 2: Manual Install
```bash
# Install Neural SDK v1.1.0 directly
pip install git+https://github.com/IntelIP/Neural-Trading-Platform.git@v1.1.0

# Verify installation
python -c "from neural_sdk import NeuralSDK, __version__; print(f'✅ Neural SDK {__version__}')"
```

## 🔑 Authentication Setup

### GitHub Access (Required)
```bash
# Option 1: Use existing GitHub credentials
git config --global credential.helper store

# Option 2: SSH keys (recommended)
ssh-keygen -t ed25519 -C "your-email@company.com"
# Add ~/.ssh/id_ed25519.pub to your GitHub account
```

### API Keys (Required for Trading)
```bash
# Create .env file
cat > .env << EOF
KALSHI_API_KEY_ID=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here
KALSHI_ENVIRONMENT=production
EOF
```

## 🏈 Quick Test - NFL WebSocket Streaming

```python
# test_installation.py
import asyncio
from neural_sdk import NeuralSDK

async def test_websocket():
    sdk = NeuralSDK.from_env()
    
    # Create WebSocket connection
    websocket = sdk.create_websocket()
    
    @websocket.on_market_data
    async def handle_data(data):
        print(f"🔴 LIVE: {data.ticker} = ${data.yes_price}")
    
    print("🔌 Connecting to Kalshi WebSocket...")
    await websocket.connect()
    
    print("📡 Subscribing to NFL markets...")
    await websocket.subscribe_markets(['KXNFLGAME*'])
    
    print("✅ WebSocket streaming active!")
    print("Press Ctrl+C to stop")
    
    try:
        await websocket.run_forever()
    except KeyboardInterrupt:
        print("👋 Stopping...")
        await websocket.disconnect()

if __name__ == "__main__":
    asyncio.run(test_websocket())
```

```bash
# Run test
python test_installation.py
```

## 🐳 Docker Setup (Optional)

```bash
# Use team Docker image
docker run -it \
  -e KALSHI_API_KEY_ID=your_key \
  -e KALSHI_API_SECRET=your_secret \
  neural-sdk:1.1.0
```

## 🔧 Development Setup

```bash
# For SDK development
git clone https://github.com/IntelIP/Neural-Trading-Platform.git
cd Neural-Trading-Platform
pip install -e .

# Run all tests
pytest tests/unit/test_websocket_simple.py -v
```

## 🆘 Troubleshooting

### Authentication Issues
```bash
# Check GitHub access
git ls-remote https://github.com/IntelIP/Neural-Trading-Platform.git

# Update credentials
git config --global --unset credential.helper
git config --global credential.helper store
```

### Import Errors
```bash
# Reinstall with force
pip uninstall neural-sdk -y
pip install --force-reinstall git+https://github.com/IntelIP/Neural-Trading-Platform.git@v1.1.0
```

### WebSocket Connection Issues
```bash
# Test API credentials
python -c "
import os
print('API Key ID:', os.getenv('KALSHI_API_KEY_ID', 'NOT SET'))
print('Environment:', os.getenv('KALSHI_ENVIRONMENT', 'development'))
"
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/IntelIP/Neural-Trading-Platform/issues)
- **Documentation**: [WebSocket Guide](WEBSOCKET_STREAMING_GUIDE.md)
- **Examples**: [examples/](../examples/) directory

---

**Ready to build real-time trading strategies!** 🎯