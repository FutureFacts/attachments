# Processors

Processors are complete file-to-LLM pipelines that handle specific file types or processing modes. They combine loading, presenting, and refining steps into a single, optimized workflow.

## Report Processor

**Module**: `attachments.pipelines.report_processor`

### `report_to_llm(att: Attachment) -> Attachment`

Generates detailed file reports with character and line counts for directories and projects.

**Matcher**: `report_match(att: Attachment) -> bool`
- Triggers when `[mode:report]` or `[format:report]` is specified
- Only processes directories (not individual files)

**Features**:
- Total statistics (file count, character count, line count)
- Detailed file table sorted by size
- File type summary with percentages and averages
- Respects ignore patterns and file filtering

**Usage**:
```python
# Basic directory report
result = Attachments("src/[mode:report]")

# Detailed file analysis
result = Attachments("src/[mode:report][files:true][force:true]")
```

**DSL Commands**:
- `mode:report` or `format:report` - Activates report mode
- `files:true` - Include individual file analysis (recommended)
- `force:true` - Process large directories (use with caution)
- `ignore:pattern` - Exclude files/directories from analysis
- `max_files:N` - Limit number of files processed

**Output Format**:
```
📊 File Report
============================================================

Total: X files, Y characters, Z lines

Characters |    Lines | Extension | File Path
------------------------------------------------------------
    XX,XXX |   X,XXX |      .ext | filename.ext
    ...

Summary by file type:
--------------------------------------------------
     .ext: N files, XXX,XXX chars, X,XXX lines (avg: XXXc/XXl) XX.X%
     ...
```

**See Also**: {doc}`../examples/project_analysis_report`

---

## Other Processors

The library includes many other specialized processors for different file types:

- `ipynb_to_llm` - Jupyter Notebooks
- `pdf_to_llm` - PDF documents  
- `pptx_to_llm` - PowerPoint presentations
- `docx_to_llm` - Word documents
- `excel_to_llm` - Excel spreadsheets
- `csv_to_llm` - CSV data files
- `image_to_llm` - Images (PNG, JPEG, etc.)
- `svg_to_llm` - SVG vector graphics
- `webpage_to_llm` - Web pages and HTML

Each processor is automatically triggered based on file type or DSL commands, providing optimized processing pipelines for different content types. 