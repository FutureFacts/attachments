# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.17.2] - 2025-01-30

### 🐛 Major Bug Fixes

- **Critical DSL Parsing Fix**: Completely rewrote `_parse_attachy()` method to properly handle DSL commands anywhere in strings
  - **Fixed command placement**: DSL commands now work when placed before paths: `[force:true][ignore:packages][files:true]src/`
  - **Improved regex patterns**: Changed from end-anchored (`$`) to global matching with `finditer()` for comprehensive command extraction
  - **Better path cleaning**: Proper removal of commands from strings using `sub()` method with clean final path resolution
  - **Enhanced shorthand support**: Improved handling of page shorthand patterns like `[1,3-5,-1]`

- **Enhanced Ignore Pattern System**: Transformed backwards ignore logic into intuitive, comprehensive file filtering
  - **Fixed custom pattern behavior**: Custom patterns like `[ignore:packages]` now ADD to essential patterns instead of replacing them
  - **Essential patterns protection**: Critical patterns (`.git`, `node_modules`, `__pycache__`, lock files) always excluded by default
  - **Flag system implementation**: New control flags for advanced users:
    - **`raw` flag**: Use only specified patterns, no essentials (`[ignore:packages,raw]`)
    - **`none` flag**: Use auto-detection (`.gitignore` or standard) (`[ignore:none]`)
    - **`raw,none` flag**: Truly ignore nothing - dangerous but available (`[ignore:raw,none]`)

### ✨ Improvements

- **Comprehensive Standard Patterns**: Updated ignore patterns to handle modern development workflows
  - **Lock files**: Added `pnpm-lock.yaml`, `package-lock.json`, `yarn.lock`, `Cargo.lock`, `poetry.lock`, `Pipfile.lock`
  - **Build directories**: Added `release`, `out` (in addition to existing `dist`, `build`, `target`)
  - **Additional patterns**: `tmp`, `temp`, `*.swp`, `*.swo`, `vendor`, `bower_components`
  - **Rust-specific**: Enhanced patterns like `target/*`, `**/target/*`

- **Intuitive Default Behavior**: Safe defaults with power-user escape hatches
  - **Default**: Essential patterns + custom patterns (safe for LLM processing)
  - **Custom patterns**: Additive behavior - adds patterns without removing essentials
  - **Expert mode**: Full control available with flag combinations

### 🔧 Technical Changes

- **DSL Parser Rewrite**: Complete overhaul of command parsing logic
  - **Global command detection**: Commands found anywhere in string, not just at end
  - **Robust pattern matching**: Handles complex command combinations reliably
  - **Improved error handling**: Better detection of malformed command syntax
  - **Order preservation**: Commands processed in order found for predictable behavior

- **Ignore Pattern Architecture**: New layered approach to file filtering
  - **Essential patterns**: 36 core patterns that protect against massive/binary files
  - **Standard patterns**: 55 comprehensive patterns for typical development
  - **Custom patterns**: User additions that enhance rather than replace defaults
  - **Flag system**: Granular control for advanced use cases

### 📊 Performance Results

Testing on real codebases shows dramatic improvements:
- **No ignore specified**: 411,175 chars (66 files) - Clean, essential filtering
- **`[ignore:none]`**: 3,884,276 chars (965 files) - Respects project `.gitignore`  
- **`[ignore:packages]`**: 319,876 chars (41 files) - Essentials + custom filtering
- **`[ignore:packages,raw]`**: 10,232,664 chars (1001 files) - Only excludes packages
- **`[ignore:raw,none]`**: 4,259,840 chars (1001 files) - Includes everything

### ⚠️ Breaking Changes

None - All changes maintain backward compatibility while fixing broken functionality.

### 🔄 Migration Guide

**DSL Usage**: Commands now work in natural positions:

```python
# Before: Only worked at the end
result = attach("src/[force:true][ignore:packages]")

# After: Works anywhere (both examples work)
result = attach("[force:true][ignore:packages]src/")  # ✅ Now works
result = attach("src/[force:true][ignore:packages]")  # ✅ Still works
```

**Ignore Patterns**: Behavior is now intuitive and additive:

```python
# Safe defaults - essentials + your patterns
attach("[ignore:packages]src/")  # Excludes .git, node_modules, AND packages

# Power user mode - exact control
attach("[ignore:packages,raw]src/")  # Excludes ONLY packages

# Respect project settings
attach("[ignore:none]src/")  # Uses .gitignore or standard patterns
```

## [0.10.0] - 2025-01-30

### 🚀 Major Features

- **Enhanced Type Dispatch System**: Complete overhaul of type matching for presenters, modifiers, and splitters
  - **Fixed PIL Image Loading**: Resolved critical issue where PIL Image objects (PngImageFile, JpegImageFile, etc.) weren't being processed by image presenters
  - **Inheritance-Based Matching**: Proper `isinstance()` checking for type annotations like `'PIL.Image.Image'`
  - **Regex Pattern Support**: Contributors can now use regex patterns in type annotations: `r'.*ImageFile$'`
  - **Smart Pattern Detection**: Automatically distinguishes between module paths (`'PIL.Image.Image'`) and regex patterns (`r'.*ImageFile$'`)

### ✨ Improvements

- **Robust Image Processing**: All image formats now work correctly with the type dispatch system
  - PNG, JPEG, GIF, BMP, WEBP images properly converted to base64
  - HEIC/HEIF support with `pillow-heif` dependency
  - SVG rendering with cairosvg/playwright fallback
  - Consistent inheritance checking across all PIL Image subclasses

- **Multiple Matching Strategies**: Enhanced type dispatch with fallback chain
  1. Exact full module.class match (`PIL.PngImagePlugin.PngImageFile == 'PIL.PngImagePlugin.PngImageFile'`)
  2. Class name match (`PngImageFile == 'Image'`)  
  3. Inheritance checking (`isinstance(obj, PIL.Image.Image)`)
  4. Regex pattern matching (`r'.*ImageFile$'`)

- **Developer Experience**: Better error handling and meaningful function wrapping
  - Fixed `@wraps` decorator to use meaningful handlers instead of fallback functions
  - Clear debug capabilities for troubleshooting type dispatch issues
  - Automatic detection of regex vs. normal type annotations

### 🐛 Bug Fixes

- **Critical Image Loading Fix**: Resolved [GitHub issue #6](https://github.com/MaximeRivest/attachments/issues/6)
  - Fixed `present.images` not working with PIL Image objects
  - Corrected type dispatch logic that was incorrectly treating `'PIL.Image.Image'` as regex pattern
  - Restored proper inheritance checking for PIL Image subclasses
  - All image tests now pass: PNG, HEIC, SVG, and multiple image processing

- **Type Annotation Processing**: Fixed regex pattern detection logic
  - Normal module paths like `'PIL.Image.Image'` no longer treated as regex
  - Explicit regex patterns (`r'pattern'` or containing metacharacters) properly detected
  - Improved `_looks_like_module_path()` heuristic for better pattern recognition

### 🔧 Technical Changes

- **Enhanced VerbNamespace**: Improved dispatch wrapper creation
  - Better function wrapping with meaningful handlers for debugging
  - Proper inheritance checking with dynamic imports
  - Support for both string type annotations and actual type objects

- **Flexible Type Matching**: Contributors can choose the best approach
  - **Inheritance approach** (recommended): `pil_image: 'PIL.Image.Image'`
  - **Regex approach** (advanced): `pil_image: r'.*ImageFile$'`
  - **No core modifications needed**: Just add functions with appropriate type annotations

### 📚 Documentation

- **Type Dispatch Examples**: Clear examples of both inheritance and regex approaches
- **Python Inheritance Tutorial**: Detailed explanation of how inheritance works in the attachments context
- **Contributor Guidelines**: Best practices for type annotations and when to use each approach

### ⚠️ Breaking Changes

None - All changes are backward compatible and enhance existing functionality.

### 🔄 Migration Guide

**For Contributors**: Enhanced type annotation options:

```python
# Inheritance approach (recommended) - works automatically
@presenter
def images(att: Attachment, pil_image: 'PIL.Image.Image') -> Attachment:
    # Matches PngImageFile, JpegImageFile, etc. via isinstance()
    pass

# Regex approach (advanced) - for complex patterns  
@presenter
def images(att: Attachment, pil_image: r'.*ImageFile$') -> Attachment:
    # Matches any class ending with 'ImageFile'
    pass
```

**For Users**: No changes needed - all image processing now works correctly out of the box.

## [0.9.0] - 2025-01-29

### 🚀 Major Features

- **Complete DSPy Integration Overhaul**: Fixed critical recursion bug and enhanced type annotation support
  - Fixed infinite recursion issue causing 56M+ character text outputs with repeated `<DSPY_TEXT_START>` markers
  - Implemented proper Pydantic core schema protocol (`__get_pydantic_core_schema__`) for seamless type annotations
  - `Attachments` objects now work directly as type annotations in DSPy signatures: `document: Attachments = dspy.InputField()`
  - Zero adapter calls needed - DSPy framework automatically calls `serialize_model()` when needed

- **In-Memory Content Processing**: Elegant data URL approach for dynamic content workflows
  - Process AI-generated content without disk I/O using `data:image/svg+xml;base64,{content}` URLs
  - Complete cycle: Load → AI Analysis → AI Generation → In-Memory Reload → Comparison
  - Perfect for iterative AI-powered content improvement workflows

### ✨ Improvements

- **Enhanced Demo**: New comprehensive "Vector Graphics and LLMs" tutorial
  - Demonstrates full AI workflow: SVG analysis → improvement generation → in-memory reload
  - Shows elegant DSPy signature usage with `Attachments` type annotations
  - R vignette style literate programming with detailed explanations
  - Uses latest OpenAI models (o3, gpt-4.1-nano) for optimal performance

- **Robust DSPy Architecture**: Clean separation between normal usage and DSPy integration
  - Normal `.text` and `.images` access works exactly like regular Attachments
  - DSPy-specific methods (`serialize_model()`, `model_dump()`) only activated when needed
  - Proper error handling when DSPy dependencies are missing
  - Duck-typing compatibility with DSPy BaseType protocol

### 🐛 Bug Fixes

- **Critical DSPy Recursion Fix**: Resolved the massive text duplication issue
  - Fixed `__str__` method that was causing infinite recursion in DSPy contexts
  - Normal text access now returns expected content length (17,894 chars vs 56M+)
  - Proper DSPy serialization only occurs when objects are passed to DSPy signatures

- **Pydantic Schema Compatibility**: Fixed schema generation errors for DSPy signatures
  - Implemented correct `pydantic_core.core_schema.plain_serializer_function_ser_schema()` usage
  - Supports both validation (string → Attachments) and serialization (Attachments → string)
  - Compatible with both Pydantic v1 and v2 APIs

### 📚 Documentation

- **Enhanced Vector Graphics Tutorial**: Complete end-to-end demonstration
  - Step-by-step progression from file loading to AI-powered improvement
  - Jupyter notebook compatible format with MyST/Jupytext
  - Visual comparisons between original and AI-improved content
  - Best practices for multimodal AI workflows

### 🔧 Technical Changes

- **DSPy Integration Module**: New `attachments.dspy` module for clean separation
  - `from attachments.dspy import Attachments` - DSPy-optimized version
  - `from attachments import Attachments` - Regular version (backward compatible)
  - Optional dependency handling with clear error messages
  - Factory functions: `make_dspy()` and `from_attachments()` for migration

- **Improved Error Handling**: Better error messages and graceful degradation
  - Clear guidance when DSPy dependencies are missing
  - Helpful installation instructions with both pip and uv commands
  - Warnings instead of hard failures for better developer experience

### ⚠️ Breaking Changes

None - All changes are backward compatible.

### 🔄 Migration Guide

**For DSPy Users**: Switch to the optimized import for better experience:

```python
# New recommended approach
from attachments.dspy import Attachments
import dspy

class AnalyzeDoc(dspy.Signature):
    document: Attachments = dspy.InputField()  # ✅ Now works perfectly!
    analysis: str = dspy.OutputField()

# Old approach still works
from attachments import Attachments
doc = Attachments("file.pdf").dspy()  # Still supported
```

**For Regular Users**: No changes needed - everything works exactly the same.

## [0.8.0] - 2025-01-23

### 🚀 Major Features

- **Enhanced URL Morphing Architecture**: Complete redesign of URL processing with intelligent file type detection
  - Replace hardcoded file extension lists with smart enhanced matchers
  - New `url_to_response` + `morph_to_detected_type` architecture for all URL processing
  - Enhanced matchers now check Content-Type headers, magic numbers, and file extensions automatically
  - Zero hardcoded file type lists across the entire codebase

### ✨ Improvements

- **Enhanced Attachment Class**: Added comprehensive helper methods for intelligent content analysis
  - `att.content_type` - Easy access to Content-Type headers from HTTP responses
  - `att.has_content` - Check if URL content is available
  - `att.get_magic_bytes(n)` - Cached magic number detection for binary file identification
  - `att.has_magic_signature(sigs)` - Test multiple binary signatures at once
  - `att.contains_in_content(patterns)` - Search for patterns in ZIP-based office formats
  - `att.is_likely_text()` - Intelligent text vs binary content detection
  - `att.get_text_sample(n)` - Safe text decoding with caching
  - `att.input_source` and `att.text_content` properties eliminate repetitive loader patterns

- **Enhanced @loader Decorator**: Automatic input source detection and preparation
  - Eliminates `getattr(att, '_file_content', None) or att.path` patterns from all loaders
  - Automatic handling of URL content vs file path inputs
  - Centralized error handling for missing dependencies

- **System-wide Processor Updates**: All file type processors now use the new morphing architecture
  - PDF, CSV, PPTX, DOCX, Excel, and Image processors updated
  - Consistent URL handling across all file types
  - Automatic detection works for any new loader without configuration

### 🐛 Bug Fixes

- **URL Display**: Fixed URLs showing temporary filenames instead of original URLs in headers
  - PDFs from URLs now show `https://example.com/doc.pdf` instead of `/tmp/tmpXXX.pdf`
  - Enhanced `display_url` metadata preservation across the processing pipeline

- **Binary Content Handling**: Improved binary content detection to prevent decode warnings
  - Better detection of binary vs text content from URLs
  - Prevents "replacement character" warnings when processing images and other binary files

### 📚 Documentation

- **URL Morphing Tutorial**: New comprehensive tutorial at `docs/scripts/how_to_load_and_morph.py`
  - Step-by-step progression from wrong approaches to best practices
  - Interactive Jupyter notebook version available
  - Demonstrates the evolution from hardcoded lists to intelligent detection

### 🔧 Technical Changes

- **Enhanced Matchers**: All matchers (`pdf_match`, `image_match`, etc.) now intelligently check:
  - File extensions: `att.path.endswith('.pdf')`
  - Content-Type headers: `'pdf' in att.metadata['content_type']`
  - Magic numbers: `att._file_content` starts with `b'%PDF'`
  - No configuration required - works automatically for URLs and files

- **Simplified Loaders**: 70% code reduction in loaders through automatic input preparation
  - PDF: `att._obj = pdfplumber.open(att.input_source)`
  - Office: `att._obj = Presentation(att.input_source)`
  - Images: `att._obj = Image.open(att.input_source)`
  - Text: `content = att.text_content`

### ⚠️ Breaking Changes

- **Deprecated `url_to_file`**: The old hardcoded `load.url_to_file` approach is deprecated
  - Replace with: `load.url_to_response | modify.morph_to_detected_type`
  - All processors have been updated automatically
  - Old approach will be removed in v1.0.0

### 🔄 Migration Guide

If you were using the old URL processing approach:

```python
# Old approach (deprecated)
att | load.url_to_file | load.pdf_to_pdfplumber

# New approach (recommended)  
att | load.url_to_response | modify.morph_to_detected_type | load.pdf_to_pdfplumber

# Or use processors (automatically uses new approach)
from attachments.pipelines import processors
result = processors.pdf_to_llm(attach("https://example.com/doc.pdf"))
```
