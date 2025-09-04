# Neural SDK Documentation

Professional documentation for Neural SDK powered by Mintlify.

## 🚀 Quick Start

### Install Mintlify CLI

```bash
npm install -g mintlify
```

### Run Documentation Locally

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Or use mintlify directly
mintlify dev
```

The documentation will be available at `http://localhost:3000`.

## 📚 Documentation Structure

```
docs-mintlify/
├── mint.json                 # Mintlify configuration
├── introduction.mdx          # Landing page
├── quickstart.mdx           # Getting started guide
├── api-reference/           # API documentation
│   ├── sdk/                # Core SDK classes
│   ├── streaming/           # WebSocket APIs
│   ├── backtesting/         # Backtesting APIs
│   └── trading/             # Trading APIs
├── guides/                  # How-to guides
│   ├── websocket-streaming.mdx
│   ├── nfl-markets.mdx
│   └── backtesting.mdx
├── examples/                # Code examples
│   └── *.mdx
└── reference/               # Technical reference
    └── *.mdx
```

## 🔧 Auto-Generation

Generate API documentation from code:

```bash
# Generate docs from docstrings
npm run generate

# Or run directly
python ../scripts/generate_mintlify_docs.py
```

## 🎨 Customization

### Update Configuration

Edit `mint.json` to customize:
- Colors and branding
- Navigation structure
- API endpoints
- Analytics

### Add New Pages

1. Create new `.mdx` file in appropriate directory
2. Add to navigation in `mint.json`
3. Use Mintlify components:

```mdx
<Card title="Title" icon="icon-name">
  Content
</Card>

<CodeGroup>
```python Python
# Code example
```
</CodeGroup>

## 📝 MDX Components

Available components:
- `<Card>` - Feature cards
- `<CardGroup>` - Grid of cards
- `<Tabs>` - Tabbed content
- `<CodeGroup>` - Multi-language code examples
- `<Steps>` - Step-by-step guides
- `<AccordionGroup>` - Collapsible sections
- `<ParamField>` - API parameters
- `<ResponseField>` - API responses

## 🚢 Deployment

### Deploy to Mintlify

```bash
# Deploy to production
npm run deploy

# Or
mintlify deploy
```

### GitHub Pages

```yaml
# .github/workflows/docs.yml
name: Deploy Docs

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm install -g mintlify
      - run: mintlify build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs-mintlify/.mintlify
```

## 📊 Analytics

Track documentation usage by adding to `mint.json`:

```json
{
  "analytics": {
    "gtag": {
      "measurementId": "G-XXXXXXXXXX"
    }
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Update documentation
4. Test locally with `mintlify dev`
5. Submit pull request

## 📞 Support

- [Mintlify Documentation](https://mintlify.com/docs)
- [Neural SDK Issues](https://github.com/IntelIP/Neural-Trading-Platform/issues)

---

Built with ❤️ using Mintlify