# GitHub Secrets Configuration Guide

## Overview

This guide documents all secrets required for the Kalshi Trading Agent System and how to configure them in GitHub.

## Required Secrets

### 1. Kalshi Trading Platform

#### KALSHI_API_KEY_ID
- **Description**: Your Kalshi API key identifier
- **Format**: String (UUID format)
- **Example**: `"ak_1234567890abcdef"`
- **How to obtain**: 
  1. Log in to Kalshi
  2. Navigate to Settings → API
  3. Create new API key
  4. Copy the Key ID

#### KALSHI_PRIVATE_KEY
- **Description**: RSA private key for Kalshi API authentication
- **Format**: Multi-line PEM format
- **Example**:
  ```
  -----BEGIN PRIVATE KEY-----
  MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcw...
  ... (multiple lines) ...
  -----END PRIVATE KEY-----
  ```
- **How to obtain**:
  1. Generate RSA key pair locally
  2. Upload public key to Kalshi
  3. Keep private key secure
- **Generation command**:
  ```bash
  openssl genrsa -out kalshi_private.pem 2048
  openssl rsa -in kalshi_private.pem -pubout -out kalshi_public.pem
  ```

#### KALSHI_ENVIRONMENT
- **Description**: Kalshi API environment
- **Format**: String
- **Values**: `"production"` or `"demo"`
- **Default**: `"demo"` for testing
- **Note**: Start with demo for development

### 2. AI/LLM Services

#### OPENROUTER_API_KEY
- **Description**: API key for OpenRouter LLM service
- **Format**: String
- **Example**: `"sk-or-v1-abc123..."`
- **How to obtain**:
  1. Sign up at https://openrouter.ai
  2. Navigate to API Keys
  3. Create new key
  4. Fund account for usage

#### AGENTUITY_SDK_KEY
- **Description**: SDK key for Agentuity agent platform
- **Format**: String
- **Example**: `"ag_sdk_123456..."`
- **How to obtain**:
  1. Sign up at https://agentuity.com
  2. Create new project
  3. Get SDK key from project settings

### 3. Infrastructure

#### REDIS_URL
- **Description**: Redis connection string
- **Format**: URL string
- **Examples**:
  - Local: `"redis://localhost:6379"`
  - Cloud: `"redis://user:pass@host:port/db"`
  - TLS: `"rediss://user:pass@host:port/db"`
- **Providers**:
  - Redis Cloud: https://redis.com/cloud
  - Upstash: https://upstash.com
  - AWS ElastiCache

#### DATABASE_URL
- **Description**: PostgreSQL connection string
- **Format**: URL string
- **Example**: `"postgresql://user:pass@host:5432/dbname"`
- **Providers**:
  - Supabase: https://supabase.com
  - Neon: https://neon.tech
  - AWS RDS

### 4. Optional Deployment Secrets

#### DOCKER_REGISTRY
- **Description**: Docker registry URL
- **Format**: URL string
- **Examples**:
  - Docker Hub: `"docker.io"`
  - GitHub: `"ghcr.io"`
  - AWS ECR: `"123456789.dkr.ecr.region.amazonaws.com"`

#### DOCKER_USERNAME
- **Description**: Docker registry username
- **Format**: String
- **Note**: For GitHub Container Registry, use your GitHub username

#### DOCKER_PASSWORD
- **Description**: Docker registry password/token
- **Format**: String
- **Note**: For GitHub, use a Personal Access Token with `write:packages` scope

#### SLACK_WEBHOOK_URL
- **Description**: Slack webhook for notifications
- **Format**: URL string
- **Example**: `"https://hooks.slack.com/services/T00/B00/XXX"`
- **How to obtain**:
  1. Go to Slack App Directory
  2. Add "Incoming Webhooks"
  3. Choose channel
  4. Copy webhook URL

## Setting Up Secrets in GitHub

### Via GitHub Web UI

1. Navigate to your repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Enter secret name and value
5. Click **Add secret**

### Via GitHub CLI

```bash
# Install GitHub CLI
brew install gh  # macOS
# or
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg

# Authenticate
gh auth login

# Add single-line secret
gh secret set KALSHI_API_KEY_ID --body "your-api-key-id"

# Add multi-line secret from file
gh secret set KALSHI_PRIVATE_KEY < kalshi_private.pem

# Add from environment variable
echo $OPENROUTER_API_KEY | gh secret set OPENROUTER_API_KEY

# List all secrets
gh secret list
```

### Via GitHub API

```bash
# Encrypt secret value
# First, get the repository public key
curl -H "Authorization: token YOUR_PAT" \
  https://api.github.com/repos/OWNER/REPO/actions/secrets/public-key

# Then encrypt and upload (requires sodium encryption)
# Use GitHub CLI instead for simplicity
```

## Environment-Specific Secrets

### Development Environment
```yaml
KALSHI_ENVIRONMENT: "demo"
REDIS_URL: "redis://localhost:6379"
DATABASE_URL: "postgresql://dev:dev@localhost:5432/kalshi_dev"
LOG_LEVEL: "DEBUG"
```

### Staging Environment
```yaml
KALSHI_ENVIRONMENT: "demo"
REDIS_URL: "redis://staging-redis:6379"
DATABASE_URL: "postgresql://user:pass@staging-db:5432/kalshi_staging"
LOG_LEVEL: "INFO"
SENTRY_DSN: "https://xxx@sentry.io/xxx"
```

### Production Environment
```yaml
KALSHI_ENVIRONMENT: "production"
REDIS_URL: "rediss://prod-redis:6380"
DATABASE_URL: "postgresql://user:pass@prod-db:5432/kalshi_prod"
LOG_LEVEL: "WARNING"
SENTRY_DSN: "https://xxx@sentry.io/xxx"
DATADOG_API_KEY: "xxx"
```

## Secret Rotation Schedule

| Secret | Rotation Frequency | Last Rotated | Next Rotation |
|--------|-------------------|--------------|---------------|
| KALSHI_API_KEY_ID | 90 days | - | - |
| KALSHI_PRIVATE_KEY | 90 days | - | - |
| OPENROUTER_API_KEY | 60 days | - | - |
| AGENTUITY_SDK_KEY | 90 days | - | - |
| DATABASE_URL | 180 days | - | - |
| DOCKER_PASSWORD | 30 days | - | - |

## Security Best Practices

### Do's
- ✅ Use GitHub Secrets for all sensitive data
- ✅ Rotate secrets regularly
- ✅ Use different secrets per environment
- ✅ Limit secret access to required workflows
- ✅ Audit secret usage regularly
- ✅ Use strong, random values
- ✅ Enable 2FA on all service accounts

### Don'ts
- ❌ Never commit secrets to repository
- ❌ Don't log secret values
- ❌ Avoid sharing secrets via email/Slack
- ❌ Don't use production secrets in development
- ❌ Never expose secrets in error messages
- ❌ Don't reuse secrets across projects

## Testing Secrets Configuration

### 1. Verify Secrets Are Set
```yaml
# .github/workflows/test-secrets.yml
name: Test Secrets
on: workflow_dispatch

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Check secrets are configured
        run: |
          secrets=(
            "KALSHI_API_KEY_ID"
            "KALSHI_PRIVATE_KEY"
            "OPENROUTER_API_KEY"
            "AGENTUITY_SDK_KEY"
            "REDIS_URL"
          )
          
          for secret in "${secrets[@]}"; do
            if [ -z "${!secret}" ]; then
              echo "❌ $secret is not set"
              exit 1
            else
              echo "✅ $secret is configured"
            fi
          done
        env:
          KALSHI_API_KEY_ID: ${{ secrets.KALSHI_API_KEY_ID }}
          KALSHI_PRIVATE_KEY: ${{ secrets.KALSHI_PRIVATE_KEY }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          AGENTUITY_SDK_KEY: ${{ secrets.AGENTUITY_SDK_KEY }}
          REDIS_URL: ${{ secrets.REDIS_URL }}
```

### 2. Test Connection
```bash
# Local test script
python -c "
import os
from redis import Redis

# Test Redis connection
redis_url = os.getenv('REDIS_URL')
if redis_url:
    r = Redis.from_url(redis_url)
    r.ping()
    print('✅ Redis connection successful')
"
```

## Troubleshooting

### Secret Not Available in Workflow
- Check secret name matches exactly (case-sensitive)
- Verify secret is set at repository level
- Ensure workflow has permissions
- Check if using fork (secrets not available)

### Multi-line Secret Issues
- Use `|` for literal multi-line in YAML
- Ensure no extra whitespace
- Preserve line endings
- Use GitHub CLI for complex secrets

### Authentication Failures
- Verify secret values are correct
- Check expiration dates
- Ensure proper format (no extra quotes)
- Test locally first

## Emergency Procedures

### If Secrets Are Compromised

1. **Immediately**:
   ```bash
   # Rotate all affected secrets
   gh secret set KALSHI_API_KEY_ID --body "new-key-id"
   ```

2. **Revoke old credentials**:
   - Kalshi: Delete API key in dashboard
   - OpenRouter: Regenerate API key
   - Database: Change passwords

3. **Audit**:
   - Check GitHub audit log
   - Review recent deployments
   - Examine access logs

4. **Notify**:
   - Team members
   - Security team
   - Affected services

## Monitoring Secret Usage

### GitHub Audit Log
```bash
# View secret access
gh api /repos/OWNER/REPO/actions/secrets/KALSHI_API_KEY_ID/activities
```

### Workflow Run Logs
- Check Actions tab
- Review workflow runs
- Look for secret mask violations
- Monitor for unusual patterns

## Local Development

### Using .env File
```bash
# .env.local (never commit this)
KALSHI_API_KEY_ID=your-dev-key
KALSHI_PRIVATE_KEY_PATH=./keys/kalshi_private.pem
KALSHI_ENVIRONMENT=demo
OPENROUTER_API_KEY=your-dev-key
AGENTUITY_SDK_KEY=your-dev-key
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://localhost/kalshi_dev
```

### Loading Secrets
```python
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv('.env.local')

# Access secrets
kalshi_key = os.getenv('KALSHI_API_KEY_ID')
```

## Compliance Notes

- Ensure compliance with financial regulations
- Document access controls
- Maintain audit trail
- Regular security reviews
- Incident response plan

---

⚠️ **Remember**: Treat all secrets as highly sensitive. Never share, log, or commit them.