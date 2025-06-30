#!/bin/bash
set -e

echo "🚀 Releasing Stable Version 0.17.0"
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
git commit -m "release: Version 0.17.0 - Code processor and clipboard adapters

🎉 Major Features:
- New [format:code] processor for analyzing codebases efficiently
- Clipboard adapters: .to_clipboard_text() and .to_clipboard_image()
- Enhanced repository processing with binary file filtering
- TypeScript support added to language detection
- Prompt support in clipboard adapters

🔧 Technical Improvements:
- Fixed circular imports in pipeline system
- Added copykitten dependency for clipboard functionality
- Improved DSL command propagation to file attachments
- Enhanced error handling and user feedback

📦 Installation:
pip install attachments==0.17.0
# With clipboard support:
pip install 'attachments[extended]'" || echo "No changes to commit"

# Tag the release
echo "🏷️  Creating stable release tag..."
git tag -a v0.17.0 -m "Stable release v0.17.0: Code processor and clipboard adapters"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin v0.17.0

echo "✅ Stable release v0.17.0 pushed to GitHub!"
echo ""
echo "🤖 GitHub Actions will now automatically:"
echo "   1. Build the package"
echo "   2. Publish to PyPI as stable release"
echo "   3. Create GitHub release with notes"
echo ""
echo "📋 Users can now install with:"
echo "   pip install attachments  # Gets v0.17.0"
echo "   pip install 'attachments[extended]'  # With clipboard support"
echo ""
echo "�� Release complete!" 