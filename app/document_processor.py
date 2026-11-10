"""Document processing module: extracts text from PDFs, detects sections, and creates chunks."""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("relay.document_processor")

PAPER_METADATA = {
    "1706.03762v7.pdf": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": "v7",
    },
    "2410.00037v1.pdf": {
        "title": "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue",
        "arxiv_id": "2410.00037",
        "version": "v1",
    },
    "2507.07955v2.pdf": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": "v2",
    },
}

SECTION_REGEX = re.compile(
    r"^(?:[0-9]+\.?[0-9]*\s+[A-Z][A-Za-z0-9\s\-\:\,]{3,50}|Abstract|Introduction|Conclusion|References|Discussion)",
    re.MULTILINE,
)


def extract_pages_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract page text using PyMuPDF (fitz) with transparent pypdf fallback."""
    pages = []
    has_fitz = False
    try:
        import fitz  # PyMuPDF
        has_fitz = True
    except ImportError:
        has_fitz = False

    if has_fitz:
        try:
            doc = fitz.open(str(pdf_path))
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text")
                pages.append({"page": page_idx + 1, "text": text})
            doc.close()
            return pages
        except Exception as e:
            logger.warning(f"PyMuPDF failed on {pdf_path}: {e}. Falling back to pypdf.")

    # Fallback to pypdf
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": page_idx + 1, "text": text})

    return pages


def detect_sections(text: str) -> List[str]:
    """Find probable section headers in the extracted text."""
    matches = SECTION_REGEX.findall(text)
    return [m.strip() for m in matches if len(m.strip()) > 3]


def tokenize_approx(text: str) -> List[str]:
    """Approximate word/subword tokens via whitespace and punctuation boundaries."""
    return text.split()


def chunk_text(
    text: str,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> List[str]:
    """Split text into overlapping chunks of approx target_tokens (using word count approx 0.75 tokens/word)."""
    words = text.split()
    target_words = int(target_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)

    if len(words) <= target_words:
        return [text] if text.strip() else []

    chunks = []
    step = max(1, target_words - overlap_words)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + target_words]
        if not chunk_words:
            break
        chunk_str = " ".join(chunk_words).strip()
        if len(chunk_str) > 40:  # avoid empty or tiny slivers
            chunks.append(chunk_str)
        if i + target_words >= len(words):
            break

    return chunks


class DocumentProcessor:
    def __init__(self, papers_dir: Path, processed_dir: Path):
        self.papers_dir = papers_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_all_papers(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        force_rebuild: bool = False,
        config_name: str = "B",
    ) -> List[Dict[str, Any]]:
        """Process all 3 research papers and persist chunks metadata."""
        out_file = self.processed_dir / f"chunks_{config_name}.json"
        if not force_rebuild and out_file.exists():
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                logger.info(f"Loaded {len(chunks)} pre-processed chunks from {out_file}")
                return chunks
            except Exception as e:
                logger.warning(f"Could not load cached chunks from {out_file}: {e}")

        all_chunks = []
        chunk_id = 0

        for filename, meta in PAPER_METADATA.items():
            pdf_path = self.papers_dir / filename
            if not pdf_path.exists():
                logger.error(f"Paper PDF not found: {pdf_path}")
                continue

            pages = extract_pages_from_pdf(pdf_path)
            current_section = "Introduction / Front Matter"

            for page_info in pages:
                page_num = page_info["page"]
                page_text = page_info["text"]

                # Check if a new section begins on this page
                sections = detect_sections(page_text)
                if sections:
                    current_section = sections[0]

                text_chunks = chunk_text(page_text, target_tokens=chunk_size, overlap_tokens=chunk_overlap)
                for chunk_str in text_chunks:
                    chunk_id += 1
                    all_chunks.append({
                        "chunk_id": f"{meta['arxiv_id']}_c{chunk_id}",
                        "paper_title": meta["title"],
                        "arxiv_id": meta["arxiv_id"],
                        "version": meta["version"],
                        "page": page_num,
                        "section": current_section,
                        "text": chunk_str,
                        "token_count_approx": int(len(chunk_str.split()) / 0.75),
                    })

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

        logger.info(f"Processed and saved {len(all_chunks)} chunks for config {config_name} to {out_file}")
        return all_chunks

