#!/bin/bash
# Deploy searchHFforMCP to public GitHub repository

set -e

SPACE_NAME="searchHFforMCP"
GITHUB_ORG="Ckal"
REPO_NAME="$SPACE_NAME"

echo "Deploying $SPACE_NAME to GitHub..."

# Add public repo as remote if not exists
if ! git remote get-url "public-$SPACE_NAME" &>/dev/null; then
    git remote add "public-$SPACE_NAME" "https://github.com/$GITHUB_ORG/$REPO_NAME.git"
fi

# Push using git subtree
git subtree push --prefix="apps/huggingface/$SPACE_NAME" "public-$SPACE_NAME" main

echo "Deployed successfully to https://github.com/$GITHUB_ORG/$REPO_NAME"
