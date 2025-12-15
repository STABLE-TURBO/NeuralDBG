#!/bin/bash
# Script to remove scope creep modules for v0.4.0 refocusing
# Run this script to complete the directory removal

echo "Removing scope creep directories for v0.4.0..."

# Already removed:
# - neural/cost/
# - neural/monitoring/
# - neural/profiling/
# - neural/docgen/

# Remove remaining tracked directories
git rm -rf neural/teams
git rm -rf neural/mlops  
git rm -rf neural/data
git rm -rf neural/config
git rm -rf neural/education
git rm -rf neural/plugins
git rm -rf neural/explainability

# Check for untracked directories and remove them
for dir in neural/marketplace neural/collaboration neural/federated; do
    if [ -d "$dir" ]; then
        echo "Removing untracked directory: $dir"
        rm -rf "$dir"
    fi
done

echo "Scope creep removal complete!"
echo "Run 'git status' to see the changes."
