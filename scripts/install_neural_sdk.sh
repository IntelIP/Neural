#!/bin/bash
# Neural SDK Installation Script
# Version: 1.4.0

echo "🧠 Installing Neural SDK v1.4.0..."
echo ""

# Check if uv is available (faster package manager)
if command -v uv &> /dev/null; then
    echo "⚡ Using uv for fast installation..."
    uv add neural-sdk
else
    echo "📦 Using pip for installation..."
    pip install neural-sdk
fi

echo ""
echo "✅ Neural SDK v1.4.0 installed successfully!"
echo "🔥 WebSocket streaming ready to use!"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python -c "
try:
    from neural_sdk import NeuralSDK, __version__
    print(f'✅ Neural SDK {__version__} imported successfully')
    print('🚀 Ready to build real-time trading strategies!')
    print('')
    print('Quick test:')
    sdk = NeuralSDK.from_env()
    websocket = sdk.create_websocket()
    print('✅ WebSocket client created successfully')
    print('🏈 NFL streaming ready')
except Exception as e:
    print(f'❌ Installation verification failed: {e}')
    print('Please check your GitHub access and try again.')
    exit(1)
"

echo ""
echo "🎯 Next Steps:"
echo "1. Set up your environment variables (KALSHI_API_KEY_ID, KALSHI_API_SECRET)"
echo "2. Check out examples/nfl_websocket_streaming.py"
echo "3. Read docs/WEBSOCKET_STREAMING_GUIDE.md"
echo ""
echo "Happy trading! 📈"