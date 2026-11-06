#!/bin/bash
# scripts/sync-to-public.sh
# Syncs filtered changes from private to public repo

set -e

# Configuration
PUBLIC_REMOTE="public"
PRIVATE_BRANCH="main"
PUBLIC_BRANCH="main"
PATHS_FILE="git-scripts/public-paths.txt"  # File with paths to include

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Syncing Private → Public ===${NC}"

# Find the git repository root - the script is in auto-lca/git-scripts/, so find auto-lca repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=""

# The script is in auto-lca/git-scripts/, so the repo root should be the parent of git-scripts/
if [ -d "$SCRIPT_DIR/../.git" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -d "auto-lca/.git" ]; then
    # We're in parent directory, auto-lca is a subdirectory
    REPO_ROOT="$(cd auto-lca && pwd)"
elif git rev-parse --git-dir > /dev/null 2>&1; then
    # We're already in a git repo, check if it has the git-scripts directory
    POTENTIAL_ROOT="$(git rev-parse --show-toplevel)"
    if [ -f "$POTENTIAL_ROOT/git-scripts/public-paths.txt" ]; then
        REPO_ROOT="$POTENTIAL_ROOT"
    else
        echo -e "${RED}Error: Found git repo but it doesn't contain git-scripts/public-paths.txt${NC}"
        echo "Please run this script from the auto-lca repository"
        exit 1
    fi
else
    echo -e "${RED}Error: Could not find auto-lca git repository${NC}"
    echo "Please run this script from the auto-lca directory or its parent"
    exit 1
fi

# Change to repo root
cd "$REPO_ROOT"
echo -e "${GREEN}Working in: $REPO_ROOT${NC}"

# Check if public remote exists, if not prompt to add it
if ! git remote get-url "$PUBLIC_REMOTE" > /dev/null 2>&1; then
    echo -e "${YELLOW}Public remote '$PUBLIC_REMOTE' not found.${NC}"
    read -p "Enter public repository URL (or press Enter to skip): " PUBLIC_URL
    if [ -n "$PUBLIC_URL" ]; then
        git remote add "$PUBLIC_REMOTE" "$PUBLIC_URL"
        echo -e "${GREEN}Added remote '$PUBLIC_REMOTE'${NC}"
    else
        echo -e "${RED}Public remote is required. Exiting.${NC}"
        echo "Add it manually with: git remote add $PUBLIC_REMOTE <public-repo-url>"
        exit 1
    fi
fi

# Check for private remote (origin) and fetch latest changes
PRIVATE_REMOTE="origin"
if ! git remote get-url "$PRIVATE_REMOTE" > /dev/null 2>&1; then
    echo -e "${YELLOW}Private remote '$PRIVATE_REMOTE' not found.${NC}"
    echo -e "${YELLOW}To get the latest changes from your private repository, we need to add it.${NC}"
    read -p "Enter private repository URL (or press Enter to use local state only): " PRIVATE_URL
    if [ -n "$PRIVATE_URL" ]; then
        git remote add "$PRIVATE_REMOTE" "$PRIVATE_URL"
        echo -e "${GREEN}Added remote '$PRIVATE_REMOTE'${NC}"
    else
        echo -e "${YELLOW}No private remote added. Will use local branch state only.${NC}"
        echo -e "${YELLOW}Note: Make sure your local branch has the latest changes!${NC}"
    fi
fi

# Ensure we're on the private branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$PRIVATE_BRANCH" ]; then
    echo -e "${YELLOW}Switching to $PRIVATE_BRANCH branch...${NC}"
    git checkout "$PRIVATE_BRANCH" || {
        echo -e "${RED}Error: Could not checkout $PRIVATE_BRANCH branch${NC}"
        echo "Current branch: $CURRENT_BRANCH"
        exit 1
    }
fi

# Fetch and reset to latest from private remote (if it exists)
if git remote get-url "$PRIVATE_REMOTE" > /dev/null 2>&1; then
    echo -e "${GREEN}Fetching latest changes from private repository ($PRIVATE_REMOTE)...${NC}"
    git fetch "$PRIVATE_REMOTE" "$PRIVATE_BRANCH" || {
        echo -e "${YELLOW}Warning: Could not fetch from $PRIVATE_REMOTE/$PRIVATE_BRANCH${NC}"
    }
    
    if git show-ref --verify --quiet refs/remotes/"$PRIVATE_REMOTE"/"$PRIVATE_BRANCH"; then
        echo -e "${GREEN}Latest commit on $PRIVATE_REMOTE/$PRIVATE_BRANCH:${NC}"
        git log --oneline "$PRIVATE_REMOTE/$PRIVATE_BRANCH" -1
        
        echo -e "${GREEN}Resetting local branch to match $PRIVATE_REMOTE/$PRIVATE_BRANCH (to get latest changes)...${NC}"
        git reset --hard "$PRIVATE_REMOTE/$PRIVATE_BRANCH" || {
            echo -e "${YELLOW}Warning: Could not reset to remote. Using local state.${NC}"
        }
        echo -e "${GREEN}Current commit after reset:${NC}"
        git log --oneline -1
    else
        echo -e "${YELLOW}Remote branch $PRIVATE_REMOTE/$PRIVATE_BRANCH not found. Using local state.${NC}"
    fi
else
    echo -e "${YELLOW}No private remote configured. Using local branch state.${NC}"
    echo -e "${GREEN}Current commit:${NC}"
    git log --oneline -1
fi

# Create a backup branch (safety)
BACKUP_BRANCH="backup-before-filter-$(date +%Y%m%d-%H%M%S)"
git branch "$BACKUP_BRANCH"
echo -e "${GREEN}Created backup branch: $BACKUP_BRANCH${NC}"

# Create a temporary branch for filtering
FILTER_BRANCH="public-filter-temp"
if git show-ref --verify --quiet refs/heads/"$FILTER_BRANCH"; then
    git branch -D "$FILTER_BRANCH"
fi
git checkout -b "$FILTER_BRANCH"

# Remove filter-repo metadata if it exists (prevents AssertionError on re-run)
if [ -d ".git/filter-repo" ]; then
    echo -e "${YELLOW}Removing previous filter-repo metadata...${NC}"
    rm -rf .git/filter-repo
fi

# Filter the repository - this rewrites ALL commit history
echo -e "${RED}WARNING: This will rewrite ALL commit history.${NC}"
echo -e "${YELLOW}Make sure public-paths.txt excludes ALL files that ever contained sensitive data!${NC}"
echo -e "${YELLOW}Files in old commits that match your patterns will be kept in history.${NC}"
read -p "Continue with filtering? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Filtering cancelled. Returning to main branch...${NC}"
    git checkout "$PRIVATE_BRANCH"
    git branch -D "$FILTER_BRANCH"
    exit 0
fi

if [ -f "$PATHS_FILE" ]; then
    echo -e "${GREEN}Filtering repository using $PATHS_FILE...${NC}"
    echo -e "${GREEN}This rewrites ALL commits in history to only include matching files.${NC}"
    # --force is needed, and this rewrites entire history
    git filter-repo --paths-from-file "$PATHS_FILE" --force
else
    echo -e "${YELLOW}Warning: $PATHS_FILE not found. Using default exclusions...${NC}"
    # Default: exclude service account keys and sensitive configs
    git filter-repo \
        --path-glob 'src/auto_lca/ist-lca-*.json' \
        --invert-paths \
        --force
fi

# CRITICAL: Create orphan branch to completely remove all commit history
# This ensures old commits with sensitive data are completely inaccessible
# We always rebuild from scratch to ensure we have the latest filtered state
echo -e "${RED}Creating orphan branch to completely remove all commit history...${NC}"
echo -e "${YELLOW}This will make the current filtered state the ONLY commit (no history).${NC}"
ORPHAN_BRANCH="public-orphan-$(date +%s)"
git checkout --orphan "$ORPHAN_BRANCH"

# Add all current files (after filtering)
git add -A

# Create a single commit with current state
git commit -m "Public release - filtered from private repository ($(date +%Y-%m-%d))"

# Delete the old filtered branch and rename orphan to the filter branch name
git branch -D "$FILTER_BRANCH" 2>/dev/null || true
git branch -m "$FILTER_BRANCH"

# Fetch public remote (optional, for info only)
echo -e "${GREEN}Checking public remote...${NC}"
git fetch "$PUBLIC_REMOTE" "$PUBLIC_BRANCH" 2>/dev/null || {
    echo -e "${YELLOW}Public branch '$PUBLIC_BRANCH' doesn't exist yet (empty repo). Will create it on push.${NC}"
}

if git show-ref --verify --quiet refs/remotes/"$PUBLIC_REMOTE"/"$PUBLIC_BRANCH"; then
    echo -e "${YELLOW}Public branch exists. It will be completely replaced with the filtered version.${NC}"
else
    echo -e "${GREEN}No existing public branch found. Will create new branch on push.${NC}"
fi

# Push to public
echo -e "${GREEN}Pushing to public repository...${NC}"
echo -e "${YELLOW}This will completely overwrite the public repository with the filtered version.${NC}"
read -p "Push to $PUBLIC_REMOTE? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Force push to completely replace public repo history
    git push "$PUBLIC_REMOTE" "$FILTER_BRANCH:$PUBLIC_BRANCH" --force
    echo -e "${GREEN}✓ Pushed to public repository${NC}"
    echo -e "${GREEN}Public repository has been completely replaced with filtered version.${NC}"
else
    echo -e "${YELLOW}Push cancelled${NC}"
fi

# Return to original branch
git checkout "$PRIVATE_BRANCH"
git branch -D "$FILTER_BRANCH"

echo -e "${GREEN}=== Sync Complete ===${NC}"
echo "Your private repo is unchanged. Public repo has been updated."
