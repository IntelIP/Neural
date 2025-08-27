# Branch Protection Rules Configuration

This document describes how to configure branch protection rules in GitHub for the Kalshi Trading Agent System.

## Main Branch Protection

### Navigate to Settings
1. Go to repository Settings
2. Navigate to Branches
3. Add rule for `main` branch

### Protection Settings

#### ✅ Require a pull request before merging
- **Required approvals**: 2
- **Dismiss stale reviews**: Yes
- **Require review from CODEOWNERS**: Yes (if configured)
- **Restrict who can dismiss reviews**: Admins only

#### ✅ Require status checks to pass
Required checks:
- `CI Pipeline / Lint & Format Check`
- `CI Pipeline / Test Suite`
- `CI Pipeline / Security Scan`
- `CI Pipeline / Build & Validate`

Settings:
- **Require branches to be up to date**: Yes
- **Status checks found in the last week**: Select all CI checks

#### ✅ Require conversation resolution
- All PR comments must be resolved

#### ✅ Require signed commits
- Enforce GPG signed commits (optional but recommended)

#### ✅ Require linear history
- Prevent merge commits

#### ✅ Include administrators
- Admins must also follow protection rules

#### ✅ Restrict who can push
- Only maintainers team
- No direct pushes (all changes via PR)

#### ❌ Do not allow:
- Force pushes
- Deletions
- Bypassing the above settings

## Develop Branch Protection

### Protection Settings

#### ✅ Require a pull request before merging
- **Required approvals**: 1
- **Dismiss stale reviews**: Yes

#### ✅ Require status checks to pass
Required checks:
- `CI Pipeline / Lint & Format Check`
- `CI Pipeline / Test Suite`

Settings:
- **Require branches to be up to date**: No (allow parallel development)

#### ✅ Require conversation resolution
- All PR comments must be resolved

#### ❌ Do not allow:
- Force pushes
- Deletions

## Release Branch Protection

For `release/*` branches:

### Protection Settings

#### ✅ Require a pull request before merging
- **Required approvals**: 2
- **Restrict who can approve**: Release managers only

#### ✅ Require status checks to pass
- All CI checks must pass

#### ✅ Restrict who can push
- Only release managers
- Only bug fixes allowed

## Setting Up Protection via GitHub CLI

```bash
# Install GitHub CLI if not already installed
brew install gh

# Authenticate
gh auth login

# Set protection for main branch
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["continuous-integration/tests","continuous-integration/lint"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true}' \
  --field restrictions='{"users":[],"teams":["maintainers"]}' \
  --field allow_force_pushes=false \
  --field allow_deletions=false

# Set protection for develop branch
gh api repos/:owner/:repo/branches/develop/protection \
  --method PUT \
  --field required_status_checks='{"strict":false,"contexts":["continuous-integration/tests"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews='{"required_approving_review_count":1}' \
  --field allow_force_pushes=false
```

## CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global owners
* @maintainer1 @maintainer2

# Component owners
/agents/ @agent-team
/data_pipeline/ @data-team
/trading_logic/ @trading-team
/agent_consumers/ @agent-team

# Critical files
/CLAUDE.md @lead-maintainer
/pyproject.toml @lead-maintainer
/.github/ @devops-team
```

## Auto-merge Configuration

For approved PRs that pass all checks:

```yaml
# .github/auto-merge.yml
name: Auto-merge

on:
  pull_request_review:
    types: [submitted]
  check_suite:
    types: [completed]

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.event.review.state == 'approved'
    steps:
      - uses: pascalgn/merge-action@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          MERGE_LABELS: "ready-to-merge,!do-not-merge"
          MERGE_METHOD: "squash"
          MERGE_COMMIT_MESSAGE: "pull-request-title"
```

## Branch Cleanup

Automatically delete merged branches:

1. Go to Settings → General
2. Under "Pull Requests", check:
   - ✅ Automatically delete head branches

## Monitoring & Compliance

### Weekly Review
- Review protection rule violations
- Check bypass attempts
- Audit direct commits (should be zero)
- Review stale branches

### Monthly Audit
```bash
# List all branches
gh api repos/:owner/:repo/branches --paginate

# Check protection status
gh api repos/:owner/:repo/branches/main/protection

# Review recent merges
gh pr list --state merged --limit 20

# Check for direct commits to protected branches
git log --oneline --graph main --not origin/main^
```

## Emergency Procedures

### Temporary Bypass (Emergency Only)
```bash
# Requires admin privileges
# Document reason in commit message
git push --force-with-lease origin main

# Immediately re-enable protection
gh api repos/:owner/:repo/branches/main/protection --method PUT
```

### Rollback Protection Changes
```bash
# View protection rules history
gh api repos/:owner/:repo/branches/main/protection

# Restore previous settings
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --input protection-backup.json
```

## Troubleshooting

### Common Issues

#### "Required status check not found"
- Ensure CI workflow has run at least once
- Check workflow file names match expected contexts
- Verify workflow triggers include PR events

#### "Push rejected due to branch protection"
- Ensure changes go through PR
- Check if you have necessary permissions
- Verify all required checks are passing

#### "Review dismissed after new commit"
- This is expected behavior
- Request new review after pushing changes
- Consider batching commits before review

## Best Practices

1. **Never disable protection** even temporarily
2. **Document any protection changes** in team notes
3. **Regular audits** of protection rules
4. **Train team** on protection rules
5. **Use draft PRs** for work in progress
6. **Automate what you can** with GitHub Actions
7. **Monitor for violations** and address quickly

## References

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [GitHub API - Branch Protection](https://docs.github.com/en/rest/branches/branch-protection)
- [GitHub CLI Documentation](https://cli.github.com/manual/)