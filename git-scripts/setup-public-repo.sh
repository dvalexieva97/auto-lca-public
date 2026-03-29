#!/bin/bash
# git-scripts/setup-public-repo.sh
# Initial setup for public/private repo split

set -e

PUBLIC_REPO_URL="$1"  # Pass public repo URL as argument
PUBLIC_REMOTE="public"

if [ -z "$PUBLIC_REPO_URL" ]; then
    echo "Usage: $0 <public-repo-url>"
    echo "Example: $0 https://github.com/username/auto-lca-public.git"
    exit 1
fi

# Add public remote
if git remote get-url "$PUBLIC_REMOTE" > /dev/null 2>&1; then
    echo "Remote '$PUBLIC_REMOTE' already exists. Updating..."
    git remote set-url "$PUBLIC_REMOTE" "$PUBLIC_REPO_URL"
else
    echo "Adding public remote..."
    git remote add "$PUBLIC_REMOTE" "$PUBLIC_REPO_URL"
fi

# Create public-paths.txt if it doesn't exist
if [ ! -f "git-scripts/public-paths.txt" ]; then
    echo "Creating git-scripts/public-paths.txt..."
    mkdir -p git-scripts
    cat > git-scripts/public-paths.txt << 'EOF'
# Files/patterns to include in public repo
src/**/*.py
*.md
*.sh
*.yaml
*.yml
*.toml
*.txt
*.sql
*.png
*.jpg
*.jpeg
Dockerfile*
Makefile
.gitignore
.github/**
requirements*.txt
pyproject.toml
setup.py
*.sql
EOF
    echo "Created git-scripts/public-paths.txt - edit it to customize what goes public"
fi

echo "Setup complete!"
echo "Now run: ./git-scripts/sync-to-public.sh"
