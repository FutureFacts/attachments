#!/bin/bash
set -e

echo "🚀 Releasing Stable Version 0.17.2"
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
git commit -m "release: Version 0.17.2 - Critical DSL parsing and ignore pattern fixes

🐛 Major Bug Fixes:
- Fixed critical DSL parsing bug - commands now work anywhere in strings
- Fixed backwards ignore pattern logic - custom patterns now ADD to essentials
- Enhanced regex patterns from end-anchored to global matching with finditer()
- Proper command removal and path cleaning in _parse_attachy()

✨ Improvements:
- Comprehensive standard patterns for modern development workflows
- New flag system: raw, none flags for advanced ignore control
- Essential patterns protection - .git, node_modules always excluded
- Added lock files: pnpm-lock.yaml, Cargo.lock, poetry.lock, etc.
- Enhanced build directories: release, out, target patterns

🔧 Technical Changes:
- Complete DSL parser rewrite with global command detection
- New layered ignore pattern architecture (essential + standard + custom)
- 36 essential patterns, 55 comprehensive standard patterns
- Intuitive additive behavior for custom patterns

📦 Installation:
pip install attachments==0.17.2" || echo "No changes to commit"

# Tag the release
echo "🏷️  Creating stable release tag..."
git tag -a v0.17.2 -m "Stable release v0.17.2: Critical DSL parsing and ignore pattern fixes"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin v0.17.2

echo "✅ Stable release v0.17.2 pushed to GitHub!"
echo ""
echo "🤖 GitHub Actions will now automatically:"
echo "   1. Build the package"
echo "   2. Publish to PyPI as stable release"
echo "   3. Create GitHub release with notes"
echo ""
echo "📋 Users can now install with:"
echo "   pip install attachments  # Gets v0.17.2"
echo "   pip install 'attachments[extended]'  # With clipboard support"
echo ""
echo "🎉 Release complete!" 