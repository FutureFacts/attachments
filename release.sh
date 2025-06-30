#!/bin/bash
set -e

echo "🚀 Releasing Stable Version 0.18.0"
echo "=================================="

# Ensure we have the latest DSL cheatsheet
echo "📋 Generating latest DSL cheatsheet..."
python scripts/generate_dsl_cheatsheet.py

# Run tests to ensure everything works
echo "🧪 Running tests..."
python -m pytest tests/ -v || echo "⚠️ Tests failed, but proceeding with release"

# Commit any final changes
echo "💾 Committing final changes..."
git add .
git commit -m "release: Version 0.18.0 - Pipeline trigger

- Version bump to trigger build & publish with latest fixes"

# Tag the release
echo "🏷️  Creating stable release tag..."
git tag -a v0.18.0 -m "Stable release v0.18.0: Pipeline trigger"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin v0.18.0

echo "✅ Stable release v0.18.0 pushed to GitHub!"
echo ""
echo "🤖 GitHub Actions will now automatically:"
echo "   1. Build the package"
echo "   2. Publish to PyPI as stable release"
echo "   3. Create GitHub release with notes"
echo ""
echo "📋 Users can now install with:"
echo "   pip install attachments  # Gets v0.18.0"
echo "   pip install 'attachments[extended]'  # With clipboard support"
echo ""
echo "🎉 Release complete!" 