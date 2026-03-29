# Public/Private Repository Sync Scripts

This directory contains scripts to help maintain separate public and private versions of the repository, with filtered commit history.

## Overview

- **Private repo**: Your main repository with all files and full history
- **Public repo**: Filtered version with sensitive files removed and rewritten history
- **One-way sync**: Changes flow from private → public only

## Setup

### 1. Initial Setup

First, create your public repository on GitHub/GitLab, then run:

```bash
auto-lca/scripts/setup-public-repo.sh https://github.com/username/auto-lca-public.git
```

This will:
- Add the public repository as a remote named `public`
- Create `git-scripts/public-paths.txt` with default file patterns to include

### 2. Customize What Goes Public

Edit `git-scripts/public-paths.txt` to specify which files/patterns should be included in the public repo. One pattern per line, supports glob patterns.

**Important**: Only files matching patterns in this file will be included in the public repo.

### 3. Install Git Hook (Optional but Recommended)

To prevent accidentally pushing filtered branches to your private repo, install the pre-push hook:

```bash
# Copy the hook template
cp git-scripts/pre-push-hook .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

Or manually create `.git/hooks/pre-push` with:

```bash
#!/bin/bash
# Prevent pushing filtered branches to private remote
protected_branches=('public-filter-temp' 'public-filtered-*')
private_remotes=('origin')

while read local_ref local_sha remote_ref remote_sha
do
    branch_name=$(git rev-parse --abbrev-ref HEAD)
    
    for protected in "${protected_branches[@]}"; do
        if [[ "$branch_name" == $protected ]]; then
            for remote in "${private_remotes[@]}"; do
                if [[ "$remote_ref" == "refs/heads/"* ]] && [[ "$remote_ref" == "refs/heads/$remote/"* ]]; then
                    echo "ERROR: Cannot push filtered branch '$branch_name' to private remote '$remote'"
                    exit 1
                fi
            done
        fi
    done
done

exit 0
```

## Usage

### Syncing to Public

When you're ready to push changes to the public repository:

```bash
./git-scripts/sync-to-public.sh
```

This script will:
1. Create a backup branch (safety)
2. Create a temporary filtered branch
3. Filter the repository using `public-paths.txt`
4. Merge with the public remote (if it exists)
5. Prompt you to push to the public repository
6. Clean up and return to your original branch

**Note**: Your private repository remains unchanged. Only the public remote is updated.

## Requirements

- `git-filter-repo` must be installed:
  ```bash
  pip install git-filter-repo
  # or
  brew install git-filter-repo  # macOS
  ```

## Important Notes

1. **History Rewriting**: The public repo will have different commit hashes than the private repo due to filtering.

2. **Never Push Public → Private**: The scripts are designed to only push to the `public` remote. Never manually push from public to private.

3. **Review Before Pushing**: The script creates a temporary branch you can inspect before pushing.

4. **Backup**: The script creates a backup branch, but consider backing up your entire repository before first use.

5. **Sensitive Files**: Make sure `public-paths.txt` doesn't include sensitive files like:
   - Service account keys (`ist-lca-*.json`)
   - API keys
   - Private configuration files
   - Any other sensitive data

## Troubleshooting

### "Public remote 'public' not found"
Run the setup script first: `./git-scripts/setup-public-repo.sh <url>`

### "git-filter-repo: command not found"
Install git-filter-repo: `pip install git-filter-repo`

### Merge conflicts
If merge conflicts occur, resolve them in the temporary branch, then manually push:
```bash
git push public public-filter-temp:main --force
```
