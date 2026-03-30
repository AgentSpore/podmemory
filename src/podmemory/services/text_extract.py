"""Text extraction from URLs and files (articles, books, PDFs, EPUBs).

Supported sources:
- Article URL → trafilatura (any website: Habr, Medium, Wikipedia, blogs)
- PDF upload → PyMuPDF (text extraction with page structure)
- EPUB upload → ebooklib + BeautifulSoup (chapter-aware extraction)
- Plain text paste → pass-through
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import trafilatura
import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from loguru import logger


@dataclass
class ExtractedText:
    text: str
    title: str
    author: str
    source_type: str  # "article", "book", "pdf", "text"
    source_url: str
    language: str
    chapters: list[dict]  # [{"title": "Chapter 1", "text": "..."}]
    word_count: int


def _split_into_chapters(text: str) -> list[dict]:
    """Split text into chapters by common heading patterns."""
    # Patterns: "Chapter 1", "Глава 1", "# Heading", "PART I", numbered sections
    chapter_re = re.compile(
        r"^(?:"
        r"#{1,3}\s+.+|"                        # Markdown headings
        r"(?:Chapter|Глава|Часть|Part)\s+[\dIVXLCDM]+[.:)?\s].*|"  # Chapter N
        r"\d+\.\s+[A-ZА-ЯЁ].{5,80}$"          # "1. Title"
        r")",
        re.MULTILINE | re.IGNORECASE,
    )

    matches = list(chapter_re.finditer(text))
    if len(matches) < 2:
        return []

    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()
        chapter_title = m.group(0).strip().lstrip("#").strip()
        if len(chapter_text) > 50:
            chapters.append({"title": chapter_title, "text": chapter_text})

    return chapters


async def extract_from_url(url: str) -> ExtractedText:
    """Extract article text from any URL via trafilatura."""
    def _extract():
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError("Failed to fetch URL. Check the link and try again.")

        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
        if not result or len(result) < 50:
            raise ValueError("Could not extract meaningful text from this URL.")

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else ""
        author = metadata.author if metadata and metadata.author else ""

        return result, title, author

    text, title, author = await asyncio.to_thread(_extract)
    chapters = _split_into_chapters(text)
    lang = "unknown"

    logger.info("Extracted article: {} ({} chars, {} chapters)", title[:50] or url[:50], len(text), len(chapters))

    return ExtractedText(
        text=text,
        title=title,
        author=author,
        source_type="article",
        source_url=url,
        language=lang,
        chapters=chapters,
        word_count=len(text.split()),
    )


async def extract_from_pdf(pdf_data: bytes, filename: str = "document.pdf") -> ExtractedText:
    """Extract text from PDF via PyMuPDF."""
    def _extract():
        import pymupdf

        doc = pymupdf.open(stream=pdf_data, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return pages

    pages = await asyncio.to_thread(_extract)
    if not pages:
        raise ValueError("PDF is empty or could not be read.")

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if len(text) < 50:
        raise ValueError("PDF contains no meaningful text (might be scanned/image-only).")

    chapters = _split_into_chapters(text)

    logger.info("Extracted PDF: {} ({} pages, {} chars, {} chapters)", filename, len(pages), len(text), len(chapters))

    return ExtractedText(
        text=text,
        title=filename.replace(".pdf", ""),
        author="",
        source_type="pdf",
        source_url="",
        language="unknown",
        chapters=chapters,
        word_count=len(text.split()),
    )


def _html_to_text(html: str) -> str:
    """Convert HTML content to clean plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def extract_from_epub(epub_data: bytes, filename: str = "book.epub") -> ExtractedText:
    """Extract text from EPUB with chapter-level structure."""
    def _extract() -> tuple[str, str, list[dict]]:
        import io

        book = epub.read_epub(io.BytesIO(epub_data))

        # Metadata
        title = ""
        author = ""
        try:
            titles = book.get_metadata("DC", "title")
            if titles:
                title = titles[0][0]
            creators = book.get_metadata("DC", "creator")
            if creators:
                author = creators[0][0]
        except Exception:
            pass

        # Extract chapters from spine order (the reading order)
        chapters: list[dict] = []
        full_parts: list[str] = []

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html = item.get_content().decode("utf-8", errors="replace")
            text = _html_to_text(html)
            if len(text.strip()) < 30:
                continue

            # Try to extract chapter title from first heading
            soup = BeautifulSoup(html, "html.parser")
            heading = soup.find(["h1", "h2", "h3"])
            chapter_title = heading.get_text(strip=True) if heading else item.get_name()

            chapters.append({"title": chapter_title, "text": text.strip()})
            full_parts.append(text.strip())

        full_text = "\n\n".join(full_parts)
        return full_text, title or filename.replace(".epub", ""), author, chapters

    full_text, title, author, chapters = await asyncio.to_thread(_extract)

    if not full_text or len(full_text) < 50:
        raise ValueError("EPUB contains no meaningful text.")

    logger.info(
        "Extracted EPUB: {} by {} ({} chars, {} chapters)",
        title[:50], author[:30] or "unknown", len(full_text), len(chapters),
    )

    return ExtractedText(
        text=full_text,
        title=title,
        author=author,
        source_type="book",
        source_url="",
        language="unknown",
        chapters=chapters,
        word_count=len(full_text.split()),
    )
