# Project Analysis with Attachments: Repo Metrics & Ignore Patterns

This guide demonstrates how to use the `Attachments` API to fetch, analyze, and report on your entire repository, leveraging powerful ignore patterns and advanced options. The focus is on practical, large-scale project analysis, with a brief section at the end introducing the `[mode:report]` feature for detailed file-level metrics.

---

## 🚀 Practical Walkthrough: Analyzing Your Repo with Attachments

The `Attachments` API lets you:
- Fetch and analyze your entire codebase or any directory
- Apply advanced ignore patterns to exclude irrelevant files
- Extract project metrics, file lists, and summaries
- Automate reporting for code health, size, and structure

### 1. Basic Repo Analysis

```python
from attachments import Attachments

# Analyze the whole repo, using standard ignore patterns for speed and relevance
report = Attachments("./[files:true][ignore:standard]")
print(report.text)
```

### 2. Custom Ignore Patterns

You can fine-tune which files and folders are included in your analysis:

```python
# Exclude node_modules, build artifacts, and test files
custom_report = Attachments("./[files:true][ignore:node_modules,dist,*.test.*,*.spec.*]")
print(custom_report.text)
```

### 3. Language or File-Type Focus

```python
# Focus on Python and Markdown files only
py_md_report = Attachments("./[files:false][glob:*.py,*.md][ignore:_build]")
print(py_md_report.text)
```

```python
# Focus on Python and Markdown files only
py_md_report = Attachments("/home/maxime/Projects/attachments/[mode:report][glob:*.py,*.md][ignore:_build]")
print(py_md_report.text)
```

### 4. Large Project Handling

```python
# Limit to the largest 100 files for a quick overview
limited_report = Attachments("./[files:true][max_files:100][ignore:standard]")
print(limited_report.text)
```

### 5. Extracting and Using Metrics

You can parse the output to extract statistics, largest files, or language breakdowns:

```python
import re

report = Attachments("./[files:true][ignore:standard]")
text = report.text

# Extract total statistics
match = re.search(r'Total: (\d+) files, ([\d,]+) characters, ([\d,]+) lines', text)
if match:
    files, chars, lines = match.groups()
    print(f"Files: {files}, Characters: {chars}, Lines: {lines}")

# Find the largest file
lines = [l for l in text.split('\n') if '|' in l and l.strip()[0].isdigit()]
if lines:
    print(f"Largest file: {lines[0]}")
```

### 6. Real-World Examples

- **Analyze a JavaScript project:**
  ```python
  js_report = Attachments("frontend/[files:true][ignore:node_modules,dist]")
  ```
- **Analyze a Python backend:**
  ```python
  py_report = Attachments("backend/[files:true][ignore:__pycache__,venv,.pytest_cache]")
  ```
- **Analyze a remote GitHub repo:**
  ```python
  remote_report = Attachments("https://github.com/user/repo.git[files:true][ignore:standard]")
  ```

---

## 🛠️ Advanced: Mastering Ignore Patterns

The `ignore` parameter is key for meaningful analysis:

- `[ignore:standard]` — Excludes common junk (node_modules, venv, build, etc.)
- `[ignore:none]` — Uses your `.gitignore` only
- `[ignore:pattern1,pattern2]` — Custom exclusions
- `[ignore:pattern,raw]` — Only your patterns (no built-ins)
- `[ignore:raw,none]` — No filtering at all

Examples:
```python
# Only your patterns
raw = Attachments("project/[files:true][ignore:*.log,raw]")
# Minimal filtering
minimal = Attachments("project/[files:true][ignore:raw,none]")
# Standard + custom
enhanced = Attachments("project/[files:true][ignore:standard,*.bak,temp]")
```

---

## 📄 Brief: The `[mode:report]` Feature

The `[mode:report]` option provides detailed file-level metrics, such as character and line counts, file type summaries, and size-ordered listings. Use it when you need a deep dive into file statistics.

### Example Usage

```python
from attachments import Attachments

# Generate a detailed report for a directory
report = Attachments("src/[mode:report][files:true][ignore:standard]")
print(report.text)
```

**Features:**
- 📊 Total statistics (file count, characters, lines)
- 📋 File table (size, extension, path)
- 📈 File type summary
- 🔍 Largest files first

---

## See Also
- DSL cheatsheet: `{doc}`../dsl_cheatsheet`
- Filtering, ignore patterns, and remote repo analysis

## Project Analysis with Report Mode

The `[mode:report]` feature generates detailed file reports with character and line counts for directories and projects. This is perfect for analyzing codebases, understanding project composition, and getting detailed metrics about your files.

## Basic Usage

### Simple Directory Report

```python
from attachments import Attachments

# Generate a basic report for a directory
report = Attachments("src/[mode:report]")
print(report.text)
```

### Detailed File Analysis

```python
# Include all files with detailed character/line analysis
report = Attachments("src/[mode:report][files:true][force:true]")
print(report.text)
```

## Report Features

The report processor provides:

- **📊 Total Statistics**: Overall file count, character count, and line count
- **📋 Detailed File Table**: Each file with its character count, line count, extension, and path
- **📈 File Type Summary**: Grouped statistics by file extension with averages and percentages
- **🔍 Size-Ordered Listing**: Files sorted by size (largest first) for easy identification of major components

## Example Output

```
📊 File Report
============================================================

Total: 39 files, 715,627 characters, 20,898 lines

Characters |    Lines | Extension | File Path
------------------------------------------------------------
   96,995 |   2,713 |      .ts | index.ts
   74,493 |   2,085 |      .ts | python-setup-manager.ts
   65,042 |   1,779 |      .ts | settings.ts
   50,772 |   1,385 |    .html | spell-book.html
   41,461 |   1,106 |    .html | settings.html
   ...

Summary by file type:
--------------------------------------------------
     .ts: 26 files,   506,948 chars,  14,687 lines (avg: 19,498c/ 564l)  70.8%
   .html:  7 files,   171,173 chars,   4,844 lines (avg: 24,453c/ 692l)  23.9%
    .css:  2 files,    19,662 chars,     803 lines (avg:  9,831c/ 401l)   2.7%
     .py:  3 files,    12,216 chars,     370 lines (avg:  4,072c/ 123l)   1.7%
     .md:  1 files,     5,628 chars,     194 lines (avg:  5,628c/ 194l)   0.8%
```

## Common Use Cases

### 1. Codebase Analysis

```python
# Analyze a TypeScript/JavaScript project
js_report = Attachments("frontend/[mode:report][files:true][ignore:node_modules,dist]")

# Analyze a Python project  
py_report = Attachments("backend/[mode:report][files:true][ignore:__pycache__,venv,.pytest_cache]")

# Analyze the entire project with smart filtering
full_report = Attachments("./[mode:report][files:true][ignore:standard]")

# Analyze a Git repository (includes commit history and branch info)
repo_report = Attachments("https://github.com/user/repo.git[mode:report][files:true]")
```

### 2. Finding Large Files

The report automatically sorts files by size, making it easy to identify:
- The largest source files that might need refactoring
- Unexpectedly large configuration or data files
- Files that dominate the codebase

### 3. Language Distribution Analysis

Use the file type summary to understand:
- What programming languages are used and their relative sizes
- Whether documentation (`.md`, `.txt`) is proportional to code
- If there are too many configuration files relative to source code

### 4. Project Health Metrics

```python
# Quick project overview
overview = Attachments("./[mode:report][files:true]")

# Parse the report to extract metrics (example)
text = overview.text
if "Total:" in text:
    total_line = [line for line in text.split('\n') if line.startswith('Total:')][0]
    print(f"Project size: {total_line}")
```

### 5. Repository Analysis with Git Information

When analyzing Git repositories, the report includes additional metadata:

```python
# Analyze a local Git repository
local_repo = Attachments("./[mode:report][files:true]")

# Analyze a remote Git repository
remote_repo = Attachments("https://github.com/microsoft/vscode.git[mode:report][files:true][ignore:standard]")

# The report will include:
# - Current branch information
# - Latest commit details (hash, author, message, date)
# - Remote URL information
# - Total commit count
# - Plus all the regular file analysis
```

## Advanced Options

### Combining with Other DSL Commands

```python
# Focus on specific file types
py_only = Attachments("src/[mode:report][files:true][glob:*.py]")

# Exclude certain directories
no_tests = Attachments("src/[mode:report][files:true][ignore:*test*,*spec*]")

# Limit file count for very large projects
limited = Attachments("src/[mode:report][files:true][max_files:100]")
```

### Mastering Ignore Patterns

The `ignore` parameter is crucial for getting meaningful reports from real projects:

```python
# Standard patterns (default) - includes git, dependencies, caches, etc.
standard = Attachments("project/[mode:report][files:true][ignore:standard]")

# Use project's .gitignore file
gitignore = Attachments("project/[mode:report][files:true][ignore:none]")

# Custom patterns for specific analysis
source_only = Attachments("project/[mode:report][files:true][ignore:*.log,*.tmp,build,dist,docs]")

# Multiple patterns
clean_analysis = Attachments("project/[mode:report][files:true][ignore:node_modules,__pycache__,*.pyc,build,dist,.git]")

# Language-specific exclusions
python_focus = Attachments("project/[mode:report][files:true][ignore:venv,env,.pytest_cache,__pycache__,.tox]")
js_focus = Attachments("project/[mode:report][files:true][ignore:node_modules,dist,build,.next,coverage]")
```

#### Advanced Ignore Flags

```python
# Raw mode - use ONLY your patterns (no built-in exclusions)
raw_patterns = Attachments("project/[mode:report][files:true][ignore:*.log,raw]")

# Combine raw with none for minimal filtering
minimal = Attachments("project/[mode:report][files:true][ignore:raw,none]")

# Standard + custom (recommended approach)
enhanced = Attachments("project/[mode:report][files:true][ignore:standard,*.bak,temp]")
```

#### Understanding Ignore Pattern Behavior

| Pattern | Behavior | Best For |
|---------|----------|-----------|
| `[ignore:standard]` | Essential exclusions + custom patterns | Most projects (safe default) |
| `[ignore:none]` | Use project's `.gitignore` | Respecting project conventions |
| `[ignore:pattern1,pattern2]` | Essential exclusions + your patterns | Custom filtering |
| `[ignore:pattern,raw]` | ONLY your patterns | Advanced control |
| `[ignore:raw,none]` | No filtering (everything included) | Complete analysis |

### Working with Large Projects

For very large projects, the report mode respects size limits:

```python
# Force processing of large directories (use with caution)
large_project = Attachments("huge_project/[mode:report][files:true][force:true]")

# Or get structure-only report for large projects
structure_only = Attachments("huge_project/[mode:report]")  # files:false is default

# Better approach: Use smart filtering for large projects
filtered_large = Attachments("huge_project/[mode:report][files:true][ignore:standard][max_files:500]")

# Clone and analyze remote repositories efficiently
remote_analysis = Attachments("https://github.com/facebook/react.git[mode:report][files:true][ignore:standard]")
```

### Git Repository Examples

```python
# Analyze popular open source projects
pytorch_report = Attachments("https://github.com/pytorch/pytorch.git[mode:report][files:true][ignore:standard][glob:*.py,*.cpp,*.h]")

# Focus on documentation and configuration
docs_report = Attachments("https://github.com/microsoft/vscode.git[mode:report][files:true][glob:*.md,*.json,*.yml,*.yaml]")

# Analyze a specific branch or commit
branch_report = Attachments("https://github.com/user/repo.git@develop[mode:report][files:true]")

# Compare different versions
v1_report = Attachments("https://github.com/user/repo.git@v1.0.0[mode:report][files:true]")
v2_report = Attachments("https://github.com/user/repo.git@v2.0.0[mode:report][files:true]")
```

## Integration Examples

### Save Report to File

```python
report = Attachments("src/[mode:report][files:true]")

# Save as markdown
with open("project_report.md", "w") as f:
    f.write(report.text)

# Or append to a larger analysis document
with open("code_analysis.md", "a") as f:
    f.write("## File Analysis\n\n")
    f.write(report.text)
    f.write("\n\n")
```

### Compare Project Versions

```python
# Compare local changes (before/after refactoring)
before = Attachments("src/[mode:report][files:true]")
# ... make changes ...
after = Attachments("src/[mode:report][files:true]")

print("=== BEFORE REFACTORING ===")
print(before.text)
print("\n=== AFTER REFACTORING ===") 
print(after.text)

# Compare Git tags/branches
main_branch = Attachments("https://github.com/user/repo.git@main[mode:report][files:true]")
dev_branch = Attachments("https://github.com/user/repo.git@develop[mode:report][files:true]")

# Compare releases
v1 = Attachments("https://github.com/user/repo.git@v1.0.0[mode:report][files:true][ignore:standard]")
v2 = Attachments("https://github.com/user/repo.git@v2.0.0[mode:report][files:true][ignore:standard]")

print(f"v1.0.0 total lines: {v1.text.split('Total: ')[1].split(' files')[0] if 'Total:' in v1.text else 'unknown'}")
print(f"v2.0.0 total lines: {v2.text.split('Total: ')[1].split(' files')[0] if 'Total:' in v2.text else 'unknown'}")
```

### Extract Specific Metrics

```python
import re

report = Attachments("src/[mode:report][files:true]")
text = report.text

# Extract total statistics
total_match = re.search(r'Total: (\d+) files, ([\d,]+) characters, ([\d,]+) lines', text)
if total_match:
    files, chars, lines = total_match.groups()
    print(f"Files: {files}, Characters: {chars}, Lines: {lines}")

# Find largest file
file_lines = [line for line in text.split('\n') if '|' in line and line.strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
if file_lines:
    largest_file = file_lines[0]  # First file is largest due to sorting
    print(f"Largest file: {largest_file}")
```

## Performance Notes

- **Directory-only reports** (`[mode:report]`) are fast and safe for any project size
- **File analysis reports** (`[mode:report][files:true]`) read all files, so respect size limits
- Use `[force:true]` carefully on large projects - it can consume significant memory
- Combine with `[ignore:...]` patterns to exclude unnecessary files for better performance
- **Git repositories** are automatically cloned to a temporary location, then analyzed
- **Remote analysis** works efficiently - only the needed files are downloaded
- Use `[ignore:standard]` as your default - it provides excellent filtering for most projects

## Real-World Examples

Here are some practical examples of analyzing popular repositories:

```python
# Analyze React.js codebase structure
react = Attachments("https://github.com/facebook/react.git[mode:report][files:true][ignore:standard][glob:*.js,*.ts]")

# Get documentation overview of a project
docs = Attachments("https://github.com/microsoft/vscode.git[mode:report][files:true][glob:*.md,README*]")

# Analyze test coverage
tests = Attachments("project/[mode:report][files:true][glob:*test*,*spec*]")

# Focus on configuration files
config = Attachments("project/[mode:report][files:true][glob:*.json,*.yml,*.yaml,*.toml,*.ini,Dockerfile]")
```

## See Also

- {doc}`../dsl_cheatsheet` - Complete DSL command reference  
- {doc}`how_to_load_and_morph` - File processing tutorial
- Directory processing with `[files:true]` vs `[files:false]`
- Ignore patterns and filtering strategies
- Git repository cloning and analysis
- Remote repository processing capabilities 