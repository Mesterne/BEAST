#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🐍 Running Black formatter..."
black .

#echo "🔍 Running Flake8 linter..."
#flake8 .


# Success check
if [ $? -eq 0 ]; then
  echo "✅ Pre-commit checks passed. Proceeding with commit."
else
  echo "❌ Pre-commit checks failed. Fix the errors before committing."
  exit 1
fi
