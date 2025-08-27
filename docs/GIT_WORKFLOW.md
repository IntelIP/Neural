# Git Workflow & Branching Strategy

## Branch Structure

```
main (production)
  ├── develop (integration)
  │   ├── feature/feature-name
  │   ├── bugfix/issue-description
  │   └── hotfix/critical-fix
  └── release/v1.0.0
```

## Branch Types

### 1. Main Branch (`main`)
- **Purpose**: Production-ready code
- **Protection**: Full protection enabled
- **Deploy**: Automatically to production
- **Merge**: Only from `release/*` or `hotfix/*` branches
- **Requirements**:
  - All tests passing
  - Code review approved
  - No merge conflicts

### 2. Develop Branch (`develop`)
- **Purpose**: Integration branch for features
- **Protection**: Requires PR and tests
- **Deploy**: To staging environment
- **Merge**: From `feature/*` and `bugfix/*` branches
- **Updated**: Daily from `main`

### 3. Feature Branches (`feature/*`)
- **Naming**: `feature/description-of-feature`
- **Created from**: `develop`
- **Merged to**: `develop`
- **Lifetime**: Until feature complete
- **Examples**:
  - `feature/add-stop-loss-monitor`
  - `feature/espn-api-integration`
  - `feature/kelly-criterion-update`

### 4. Bugfix Branches (`bugfix/*`)
- **Naming**: `bugfix/issue-number-description`
- **Created from**: `develop`
- **Merged to**: `develop`
- **Lifetime**: Until bug fixed
- **Examples**:
  - `bugfix/123-websocket-reconnection`
  - `bugfix/456-redis-timeout`

### 5. Release Branches (`release/*`)
- **Naming**: `release/vX.Y.Z`
- **Created from**: `develop`
- **Merged to**: `main` and back to `develop`
- **Purpose**: Final testing and version prep
- **Allowed changes**: Bug fixes only

### 6. Hotfix Branches (`hotfix/*`)
- **Naming**: `hotfix/critical-issue`
- **Created from**: `main`
- **Merged to**: `main` and `develop`
- **Purpose**: Emergency production fixes
- **Review**: Expedited process

## Workflow Steps

### Starting New Feature

```bash
# 1. Update develop branch
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/my-new-feature

# 3. Work on feature
git add .
git commit -m "feat: add new feature description"

# 4. Push to remote
git push -u origin feature/my-new-feature

# 5. Create Pull Request to develop
```

### Creating a Release

```bash
# 1. Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# 2. Update version numbers
# Update pyproject.toml, README.md, etc.

# 3. Final testing and fixes
git add .
git commit -m "chore: prepare release v1.0.0"

# 4. Merge to main
git checkout main
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"

# 5. Merge back to develop
git checkout develop
git merge --no-ff release/v1.0.0

# 6. Push everything
git push origin main develop --tags

# 7. Delete release branch
git branch -d release/v1.0.0
git push origin --delete release/v1.0.0
```

### Emergency Hotfix

```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug

# 2. Fix the issue
git add .
git commit -m "hotfix: fix critical production bug"

# 3. Merge to main
git checkout main
git merge --no-ff hotfix/critical-bug
git tag -a v1.0.1 -m "Hotfix version 1.0.1"

# 4. Merge to develop
git checkout develop
git merge --no-ff hotfix/critical-bug

# 5. Push and cleanup
git push origin main develop --tags
git branch -d hotfix/critical-bug
git push origin --delete hotfix/critical-bug
```

## Commit Message Convention

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or fixes
- `chore`: Build process or auxiliary tool changes
- `ci`: CI/CD changes

### Examples
```bash
feat(websocket): add automatic reconnection logic

fix(redis): handle connection timeout gracefully

docs(readme): update installation instructions

refactor(agents): simplify message processing logic

test(trading): add Kelly Criterion unit tests

chore(deps): update dependencies to latest versions
```

## Pull Request Process

### 1. Create PR
- Use PR template
- Link related issues
- Add appropriate labels
- Assign reviewers

### 2. PR Title Format
```
[TYPE] Brief description (#issue-number)
```

Examples:
- `[FEATURE] Add WebSocket reconnection logic (#123)`
- `[BUGFIX] Fix Redis timeout handling (#456)`
- `[HOTFIX] Critical trading logic fix (#789)`

### 3. PR Checklist
- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Reviewed by at least 1 person
- [ ] CI/CD checks pass

### 4. Review Process
1. **Draft PR**: For early feedback
2. **Ready for Review**: When complete
3. **Changes Requested**: Address feedback
4. **Approved**: Ready to merge
5. **Merged**: Squash and merge to target branch

## Branch Protection Rules

### Main Branch
```yaml
protection_rules:
  - require_pull_request_reviews:
      required_approving_review_count: 2
      dismiss_stale_reviews: true
  - require_status_checks:
      strict: true
      contexts:
        - continuous-integration/tests
        - continuous-integration/lint
  - enforce_admins: false
  - restrict_push:
      teams: ["maintainers"]
  - allow_force_pushes: false
  - allow_deletions: false
```

### Develop Branch
```yaml
protection_rules:
  - require_pull_request_reviews:
      required_approving_review_count: 1
  - require_status_checks:
      contexts:
        - continuous-integration/tests
  - enforce_admins: false
  - allow_force_pushes: false
```

## Versioning Strategy

### Semantic Versioning (SemVer)
Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Examples
- `1.0.0` → `2.0.0`: Breaking API changes
- `1.0.0` → `1.1.0`: New feature added
- `1.0.0` → `1.0.1`: Bug fix

### Pre-release Versions
- Alpha: `1.0.0-alpha.1`
- Beta: `1.0.0-beta.1`
- Release Candidate: `1.0.0-rc.1`

## Git Aliases (Optional)

Add to `~/.gitconfig`:

```ini
[alias]
    # Create feature branch
    feature = "!f() { git checkout develop && git pull && git checkout -b feature/$1; }; f"
    
    # Create bugfix branch
    bugfix = "!f() { git checkout develop && git pull && git checkout -b bugfix/$1; }; f"
    
    # Create release branch
    release = "!f() { git checkout develop && git pull && git checkout -b release/$1; }; f"
    
    # Create hotfix branch
    hotfix = "!f() { git checkout main && git pull && git checkout -b hotfix/$1; }; f"
    
    # Finish feature (merge to develop)
    finish-feature = "!f() { git checkout develop && git merge --no-ff $1 && git branch -d $1; }; f"
    
    # Pretty log
    lg = log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
```

## Merge Strategies

### Feature → Develop
- Strategy: **Squash and Merge**
- Keeps develop history clean
- One commit per feature

### Develop → Main
- Strategy: **Create Merge Commit**
- Preserves release history
- Clear release points

### Hotfix → Main/Develop
- Strategy: **Create Merge Commit**
- Tracks emergency fixes
- Maintains audit trail

## Conflict Resolution

### Prevention
1. Pull latest changes daily
2. Keep branches short-lived
3. Communicate with team
4. Use feature flags for long features

### Resolution Steps
```bash
# 1. Update target branch
git checkout develop
git pull origin develop

# 2. Merge into feature branch
git checkout feature/my-feature
git merge develop

# 3. Resolve conflicts
# Edit conflicted files
git add .
git commit -m "resolve: merge conflicts with develop"

# 4. Test thoroughly
pytest tests/

# 5. Push resolved branch
git push origin feature/my-feature
```

## Release Checklist

### Pre-Release
- [ ] All features merged to develop
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Security audit completed

### Release
- [ ] Create release branch
- [ ] Final testing on staging
- [ ] Merge to main
- [ ] Tag release
- [ ] Merge back to develop

### Post-Release
- [ ] Deploy to production
- [ ] Monitor for issues
- [ ] Update release notes
- [ ] Notify stakeholders
- [ ] Archive release branch

## Rollback Procedure

### Quick Revert
```bash
# Revert last commit on main
git checkout main
git revert HEAD
git push origin main
```

### Tag Rollback
```bash
# Deploy previous version
git checkout v1.0.0
git checkout -b hotfix/rollback-to-v1.0.0
# Deploy this branch
```

### Full Rollback
```bash
# Reset to previous release
git checkout main
git reset --hard v1.0.0
git push --force-with-lease origin main
```

## Best Practices

### Do's
- ✅ Pull before starting work
- ✅ Commit frequently with clear messages
- ✅ Test before pushing
- ✅ Keep branches focused
- ✅ Update documentation
- ✅ Use descriptive branch names

### Don'ts
- ❌ Commit directly to main
- ❌ Force push to shared branches
- ❌ Leave branches unmerged for weeks
- ❌ Merge without review
- ❌ Include credentials in commits
- ❌ Rewrite public history

## Tools & Integration

### Required Tools
- Git 2.30+
- GitHub CLI (`gh`)
- Pre-commit hooks

### Setup
```bash
# Install GitHub CLI
brew install gh

# Install pre-commit
pip install pre-commit
pre-commit install

# Configure git
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

### Useful Commands
```bash
# View branch graph
git log --graph --oneline --all

# Clean up local branches
git branch --merged | grep -v "\*\|main\|develop" | xargs -n 1 git branch -d

# Update fork from upstream
git remote add upstream https://github.com/original/repo.git
git fetch upstream
git checkout main
git merge upstream/main
```