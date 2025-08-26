"""OCR text extraction presenters."""

from ...core import Attachment, presenter


@presenter
def ocr(att: Attachment, pdf_reader: "pdfplumber.PDF") -> Attachment:
    """
    Extract text from scanned PDF using OCR (pytesseract).

    This presenter is useful for scanned PDFs with no extractable text.
    Requires: pip install pytesseract pillow
    Also requires tesseract binary: apt-get install tesseract-ocr (Ubuntu) or brew install tesseract (Mac)
    You can specify the language for OCR using the `lang` command, e.g., `[lang:ara]` for Arabic.
    """
    try:
        import io

        import pypdfium2 as pdfium
        import pytesseract
        from PIL import Image
    except ImportError as e:
        att.text += "\n## OCR Text Extraction\n\n"
        att.text += "⚠️ **OCR not available**: Missing dependencies.\n\n"
        att.text += "To enable OCR for scanned PDFs:\n"
        att.text += "```bash\n"
        att.text += "pip install pytesseract pypdfium2\n"
        att.text += "# Ubuntu/Debian:\n"
        att.text += "sudo apt-get install tesseract-ocr\n"
        att.text += "# For other languages (e.g., French):\n"
        att.text += "sudo apt-get install tesseract-ocr-fra\n"
        att.text += "# macOS:\n"
        att.text += "brew install tesseract\n"
        att.text += "```\n\n"
        att.text += f"Error: {e}\n\n"
        return att

    att.text += "\n## OCR Text Extraction\n\n"

    try:
        # Get PDF bytes for pypdfium2
        if "temp_pdf_path" in att.metadata:
            with open(att.metadata["temp_pdf_path"], "rb") as f:
                pdf_bytes = f.read()
        elif att.path:
            with open(att.path, "rb") as f:
                pdf_bytes = f.read()
        else:
            att.text += "⚠️ **OCR failed**: Cannot access PDF file.\n\n"
            return att

        # Open with pypdfium2
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
        num_pages = len(pdf_doc)

        # Process pages (limit for performance)
        if "selected_pages" in att.metadata:
            pages_to_process = att.metadata["selected_pages"]
        else:
            # Limit OCR to first 5 pages by default (OCR is slow)
            pages_to_process = range(1, min(6, num_pages + 1))

        total_ocr_text = ""
        successful_pages = 0

        # Get language from commands, default to English
        ocr_lang = att.commands.get("lang", "eng")

        for page_num in pages_to_process:
            if 1 <= page_num <= num_pages:
                try:
                    page = pdf_doc[page_num - 1]

                    # Render page as image
                    pil_image = page.render(scale=2).to_pil()  # Higher scale for better OCR

                    # Perform OCR
                    page_text = pytesseract.image_to_string(pil_image, lang=ocr_lang)

                    if page_text.strip():
                        att.text += f"### Page {page_num} (OCR)\n\n{page_text.strip()}\n\n"
                        total_ocr_text += page_text.strip()
                        successful_pages += 1
                    else:
                        att.text += f"### Page {page_num} (OCR)\n\n*[No text detected by OCR]*\n\n"

                except Exception as e:
                    att.text += f"### Page {page_num} (OCR)\n\n*[OCR failed: {str(e)}]*\n\n"

        # Clean up
        pdf_doc.close()

        # Add OCR summary
        att.text += "**OCR Summary**:\n"
        att.text += f"- Pages processed: {len(pages_to_process)}\n"
        att.text += f"- Language: {ocr_lang}\n"
        att.text += f"- Pages with OCR text: {successful_pages}\n"
        att.text += f"- Total OCR text length: {len(total_ocr_text)} characters\n\n"

        # Update metadata
        att.metadata.update(
            {
                "ocr_performed": True,
                "ocr_pages_processed": len(pages_to_process),
                "ocr_lang": ocr_lang,
                "ocr_pages_successful": successful_pages,
                "ocr_text_length": len(total_ocr_text),
            }
        )

    except Exception as e:
        att.text += f"⚠️ **OCR failed**: {str(e)}\n\n"
        att.metadata["ocr_error"] = str(e)

    return att
