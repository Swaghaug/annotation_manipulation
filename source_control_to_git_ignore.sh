#!/bin/bash

# Script: source_control_to_git_ignore.sh
# Purpose: Add all untracked files to the .gitignore file.

echo "Adding all untracked files to .gitignore..."

# Ensure the script is run in a Git repository
if [ ! -d .git ]; then
  echo "Error: This script must be run in the root of a Git repository."
  exit 1
fi

# Append untracked files to .gitignore
git status --untracked-files=all --short | grep '^??' | cut -c4- >> .gitignore

# Remove ignored files from Git's index (if any)
git rm -r --cached . >/dev/null 2>&1

# Verify the changes
echo "The following files have been added to .gitignore:"
git status --untracked-files=all --short | grep '^??' | cut -c4-

echo "Done. Please review the .gitignore file and commit the changes if necessary."
