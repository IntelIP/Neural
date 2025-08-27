# GitHub Push Guide

## Pre-Push Checklist

### 1. Verify Local Setup
```bash
# Check git is initialized
git status

# Verify remote is not set yet
git remote -v

# Ensure you're on main branch
git branch
```

### 2. Clean Sensitive Data
```bash
# Ensure .env is in .gitignore
grep "^\.env$" .gitignore

# Check for any secrets in code
grep -r "KALSHI_API_KEY" --exclude-dir=.git --exclude=.env* .
grep -r "OPENROUTER_API_KEY" --exclude-dir=.git --exclude=.env* .
grep -r "AGENTUITY_SDK_KEY" --exclude-dir=.git --exclude=.env* .

# Verify no credentials in committed files
git diff --cached | grep -i "api_key\|secret\|password\|token"
```

### 3. Verify Documentation
```bash
# Check all required docs exist
ls -la README.md CLAUDE.md CONTRIBUTING.md
ls -la docs/
ls -la .github/
```

## Initial Repository Setup

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository settings:
   - **Name**: `Kalshi_Agentic_Agent`
   - **Description**: "Autonomous multi-agent trading system for Kalshi sports event contracts"
   - **Visibility**: Private (initially)
   - **DO NOT** initialize with README, .gitignore, or license

### Step 2: Initialize Local Repository

```bash
# If not already initialized
git init

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/Kalshi_Agentic_Agent.git

# Verify remote
git remote -v
```

### Step 3: Prepare Initial Commit

```bash
# Stage all files
git add .

# Verify what's being staged
git status

# Check file count
git ls-files | wc -l

# Create initial commit
git commit -m "feat: initial commit - production-ready multi-agent trading system

- Complete repository structure with clear naming conventions
- Comprehensive CI/CD pipeline with GitHub Actions
- Full documentation suite (README, CLAUDE.md, CONTRIBUTING.md)
- Redis-based agent communication architecture
- Agentuity framework integration
- Kelly Criterion trading logic implementation
- WebSocket data pipeline for real-time processing
- PostgreSQL database with Alembic migrations
- Security scanning and automated testing
- Docker containerization support"
```

### Step 4: Push to GitHub

```bash
# Push main branch
git push -u origin main

# Create and push develop branch
git checkout -b develop
git push -u origin develop

# Return to main
git checkout main
```

## Configure GitHub Repository

### Step 1: Set Up Secrets

Navigate to: Settings → Secrets and variables → Actions

Add the following secrets:

#### Required Secrets
```yaml
# Kalshi Trading
KALSHI_API_KEY_ID: "your-kalshi-api-key-id"
KALSHI_PRIVATE_KEY: |
  -----BEGIN PRIVATE KEY-----
  your-private-key-content
  -----END PRIVATE KEY-----
KALSHI_ENVIRONMENT: "production"  # or "demo" for testing

# AI/LLM
OPENROUTER_API_KEY: "your-openrouter-api-key"

# Agentuity Platform
AGENTUITY_SDK_KEY: "your-agentuity-sdk-key"

# Infrastructure
REDIS_URL: "redis://your-redis-host:6379"

# Optional - for deployment
DOCKER_REGISTRY: "your-docker-registry"
DOCKER_USERNAME: "your-docker-username"
DOCKER_PASSWORD: "your-docker-password"
```

### Step 2: Configure Environments

Navigate to: Settings → Environments

Create two environments:

#### Staging Environment
- **Name**: staging
- **Protection rules**: 
  - Only from: `develop` branch
  - Required reviewers: 1
- **Secrets**: Add staging-specific overrides

#### Production Environment
- **Name**: production
- **Protection rules**:
  - Only from: `main` branch
  - Required reviewers: 2
  - Restrict deployments to specific users
- **Secrets**: Add production-specific values

### Step 3: Enable Branch Protection

Navigate to: Settings → Branches

#### Main Branch Protection
```bash
# Using GitHub CLI (if installed)
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["CI Pipeline / Lint & Format Check","CI Pipeline / Test Suite","CI Pipeline / Security Scan","CI Pipeline / Build & Validate"]}' \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true}' \
  --field enforce_admins=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

Or manually:
1. Click "Add rule"
2. Branch name pattern: `main`
3. Enable:
   - ✅ Require pull request before merging (2 approvals)
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date
   - ✅ Include administrators
   - ✅ Restrict who can push (only maintainers)

#### Develop Branch Protection
Similar to main but with 1 required approval.

### Step 4: Configure GitHub Pages (Optional)

For documentation hosting:

1. Settings → Pages
2. Source: Deploy from branch
3. Branch: main
4. Folder: /docs
5. Save

### Step 5: Set Up Teams (If Organization)

1. Settings → Manage access → Invite teams
2. Create teams:
   - `maintainers` - Admin access
   - `developers` - Write access
   - `qa-team` - Triage access

### Step 6: Configure Webhooks (Optional)

For external integrations:

1. Settings → Webhooks → Add webhook
2. Payload URL: Your monitoring service
3. Events: Push, Pull Request, Deployment

## Verify CI/CD Pipeline

### Step 1: Create Test PR

```bash
# Create feature branch
git checkout -b feature/test-ci
echo "# Test CI" > test.md
git add test.md
git commit -m "test: verify CI pipeline"
git push -u origin feature/test-ci
```

### Step 2: Open Pull Request

1. Go to repository on GitHub
2. Click "Compare & pull request"
3. Target: develop
4. Create pull request

### Step 3: Verify Checks

Ensure all checks pass:
- ✅ Lint & Format Check
- ✅ Test Suite
- ✅ Security Scan
- ✅ Build & Validate
- ✅ PR Checks

### Step 4: Clean Up

```bash
# After merge, delete test branch
git checkout develop
git pull origin develop
git branch -d feature/test-ci
git push origin --delete feature/test-ci
```

## First Deployment

### Step 1: Prepare for Deployment

```bash
# Ensure on develop branch
git checkout develop

# Tag release candidate
git tag -a v0.1.0-rc.1 -m "Release candidate 1 for v0.1.0"
git push origin v0.1.0-rc.1
```

### Step 2: Deploy to Staging

```bash
# Trigger staging deployment
# This happens automatically on push to develop

# Or manually via GitHub Actions
gh workflow run deploy.yml -f environment=staging -f version=v0.1.0-rc.1
```

### Step 3: Production Release

```bash
# Create release PR
git checkout -b release/v0.1.0
git push -u origin release/v0.1.0

# After approval and merge to main
git checkout main
git pull origin main
git tag -a v0.1.0 -m "Initial release v0.1.0"
git push origin v0.1.0

# Create GitHub Release
gh release create v0.1.0 \
  --title "v0.1.0 - Initial Release" \
  --notes "Initial production release of Kalshi Trading Agent System" \
  --target main
```

## Post-Push Tasks

### Immediate Actions

1. **Verify Repository Access**:
   ```bash
   # Clone in new directory to test
   cd /tmp
   git clone https://github.com/YOUR_USERNAME/Kalshi_Agentic_Agent.git
   cd Kalshi_Agentic_Agent
   ```

2. **Check CI Status**:
   - Go to Actions tab
   - Verify workflows are detected
   - Check for any configuration errors

3. **Update README Badge**:
   ```markdown
   ![CI Pipeline](https://github.com/YOUR_USERNAME/Kalshi_Agentic_Agent/workflows/CI%20Pipeline/badge.svg)
   ```

### Within 24 Hours

1. **Security Scan**:
   - Enable Dependabot alerts
   - Review security recommendations
   - Set up code scanning

2. **Documentation**:
   - Verify all links work
   - Check rendered markdown
   - Update any absolute paths

3. **Team Access**:
   - Invite collaborators
   - Set up CODEOWNERS file
   - Configure notifications

### Within First Week

1. **Monitoring Setup**:
   - Configure error tracking
   - Set up performance monitoring
   - Add deployment notifications

2. **Backup Strategy**:
   - Set up repository mirroring
   - Configure automated backups
   - Document recovery procedures

3. **Performance Baseline**:
   - Run initial load tests
   - Document response times
   - Set up metrics collection

## Troubleshooting

### Common Issues

#### Push Rejected - Large Files
```bash
# Check for large files
find . -type f -size +100M

# Add to .gitignore if needed
echo "large-file.bin" >> .gitignore

# Remove from git history
git filter-branch --index-filter 'git rm --cached --ignore-unmatch large-file.bin' HEAD
```

#### Push Rejected - Credentials Detected
```bash
# Remove sensitive data
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/sensitive-file' \
  --prune-empty --tag-name-filter cat -- --all
```

#### CI Workflows Not Triggering
1. Check workflow syntax
2. Verify file location (.github/workflows/)
3. Ensure proper permissions
4. Check branch protection settings

#### Permission Denied
```bash
# For HTTPS
git config --global credential.helper cache

# For SSH
ssh-add ~/.ssh/id_rsa
```

## Security Checklist

Before making repository public:

- [ ] All secrets in GitHub Secrets
- [ ] No hardcoded credentials
- [ ] .env.example has placeholder values
- [ ] Security scanning enabled
- [ ] Dependency review enabled
- [ ] Branch protection configured
- [ ] CODEOWNERS file created
- [ ] Security policy added
- [ ] Vulnerability reporting enabled

## Quick Reference

### Essential Commands
```bash
# Initial push
git push -u origin main

# Create develop branch
git checkout -b develop
git push -u origin develop

# Create feature branch
git checkout -b feature/new-feature
git push -u origin feature/new-feature

# Update from upstream
git fetch origin
git merge origin/develop

# Tag release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### GitHub CLI Commands
```bash
# Check workflow runs
gh run list

# View workflow details
gh run view

# Watch workflow in progress
gh run watch

# List issues
gh issue list

# Create issue
gh issue create --title "Bug report" --body "Description"

# List PRs
gh pr list

# Create PR
gh pr create --title "Feature" --body "Description"
```

## Next Steps

After successful push:

1. **Test the CI pipeline** with a small PR
2. **Configure deployment** to Agentuity platform
3. **Set up monitoring** and alerting
4. **Document API endpoints** if applicable
5. **Create initial issues** for known improvements
6. **Invite team members** and assign roles
7. **Schedule regular dependency updates**
8. **Plan first sprint** using GitHub Projects

---

Remember: This is a financial trading system. Ensure all security measures are properly configured before deploying to production.