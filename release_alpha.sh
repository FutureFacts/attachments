#!/bin/bash
set -e

echo "🚀 Releasing Alpha Version 0.17.0a1"
echo "====================================="

# Ensure we have the latest DSL cheatsheet
echo "📋 Generating latest DSL cheatsheet..."
python scripts/generate_dsl_cheatsheet.py

# Commit any final changes
echo "💾 Committing final changes..."
git add .
git commit -m "feat: Code processor and clipboard adapters v0.17.0a1

- New [format:code] processor for analyzing codebases efficiently
- Clipboard adapters: .to_clipboard_text() and .to_clipboard_image()
- Enhanced repository processing with binary file filtering
- TypeScript support added to language detection
- Prompt support in clipboard adapters" || echo "No changes to commit"

# Tag the release
echo "🏷️  Creating alpha release tag..."
git tag -a v0.17.0a1 -m "Alpha release v0.17.0a1: Code processor and clipboard adapters"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin v0.17.0a1

echo "✅ Alpha release v0.17.0a1 pushed to GitHub!"
echo ""
echo "🤖 GitHub Actions will now automatically:"
echo "   1. Build the package"
echo "   2. Publish to PyPI as pre-release"
echo "   3. Create GitHub release with notes"
echo ""
echo "📋 Once published, alpha testers can install with:"
echo "   pip install attachments==0.17.0a1"
echo "   # or"
echo "   pip install --pre attachments"
echo ""
echo "🛡️  Regular users still get stable version:"
echo "   pip install attachments  # Gets previous stable"

# Test the code processor
python -c "
from attachments import Attachments, set_verbose
set_verbose(False)
a = Attachments('src/attachments/adapt.py[format:code]')
print('✅ Code processor works!')
print(f'Content length: {len(a.text)}')
"

# Test clipboard (requires copykitten to be installed)
uv add copykitten  # or pip install copykitten
python -c "
from attachments import Attachments
a = Attachments('README.md')
a.to_clipboard_text('Summarize this:')
print('✅ Clipboard adapter works!')
" 