# pdf_utils.py
#
# Functionality：
#   1. Download arXiv PDF to local storage
#   2. Extract plain text using PyMuPDF
#   3. Heuristically locate chapter positions within plain text
#   4. Only Find specified chapters: Introduction / Methods / Limitations / Conclusion

import requests
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import fitz
import re


def download_pdf(pdf_url: str, save_dir: str = "pdf_cache") -> Optional[str]:
    """
    Download the PDF to your local directory using the URL.
    """
    try:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        filename = pdf_url.rstrip("/").split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        filepath = save_dir_path / filename

        if not filepath.exists():
            print(f"[PDF] Downloading: {pdf_url}")
            resp = requests.get(pdf_url, timeout=60)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
        else:
            print(f"[PDF] Using cached file: {filepath}")

        return str(filepath)
    except Exception as e:
        print(f"[Warning] Failed to download PDF from {pdf_url}: {e}")
        return None


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """
    Use PyMuPDF to extract plain text from a local PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception as e:
        print(f"[Warning] Failed to extract text from {pdf_path}: {e}")
        return ""


def get_pdf_text_from_url(
    pdf_url: str,
    save_dir: str = "pdf_cache",
    max_pages: Optional[int] = None,
) -> str:
    """
    Download and parse the PDF, returning a plain text string.
    """
    pdf_path = download_pdf(pdf_url, save_dir=save_dir)
    if not pdf_path:
        return ""
    return extract_text_from_pdf(pdf_path, max_pages=max_pages)



SECTION_PATTERNS: Dict[str, re.Pattern] = {
    # canonical_name: regex
    "introduction": re.compile(
        r"\b(?:\d+\s+)?introduction\b", re.IGNORECASE
    ),
    "related_work": re.compile(
        r"\b(?:\d+(\.\d+)*\s+)?(related work|background)\b", re.IGNORECASE
    ),
    "methods": re.compile(
        r"\b(?:\d+(\.\d+)*\s+)?(methodology|methods?|approach|proposed method|model)\b",
        re.IGNORECASE,
    ),
    "experiments": re.compile(
        r"\b(?:\d+(\.\d+)*\s+)?(experiments?|experimental results|results and discussion)\b",
        re.IGNORECASE,
    ),
    "limitations": re.compile(
        r"\b(?:\d+(\.\d+)*\s+)?(limitations?|limitation and future work)\b",
        re.IGNORECASE,
    ),
    "conclusion": re.compile(
        r"\b(?:\d+(\.\d+)*\s+)?(conclusion[s]?|concluding remarks)\b",
        re.IGNORECASE,
    ),
    "discussion": re.compile(
        r"\b(?:\d+(\.\d+)*\s+)?(discussion)\b",
        re.IGNORECASE,
    ),
}


def _find_sections_in_text(full_text: str) -> List[Dict[str, object]]:
    """
    Locate the approximate starting position of each chapter in the full text and return a list sorted by appearance order:
    [
    {"name": "introduction", "start": 123, "end": 456},
    {"name": "methods", "start": 456, "end": 999},
    ...
    ]
    """
    matches: List[Tuple[int, str]] = []

    for name, pattern in SECTION_PATTERNS.items():
        m = pattern.search(full_text)
        if m:
            matches.append((m.start(), name))

    if not matches:
        return []

    matches.sort(key=lambda x: x[0])

    sections: List[Dict[str, object]] = []
    for idx, (start_pos, name) in enumerate(matches):
        if idx + 1 < len(matches):
            end_pos = matches[idx + 1][0]
        else:
            end_pos = len(full_text)
        sections.append(
            {
                "name": name,
                "start": start_pos,
                "end": end_pos,
            }
        )

    return sections


def extract_relevant_sections_from_pdf(
    pdf_url: str,
    save_dir: str = "pdf_cache",
    max_pages: int = 12,
    include_sections: Optional[List[str]] = None,
    max_chars: int = 16000,
) -> str:
    """
        Downloads a PDF and extracts specific sections to construct a context string for LLMs.

        This function attempts to identify and extract relevant sections (e.g., Introduction,
        Methods, Limitations, Conclusion) from the PDF. If structured section extraction fails
        or yields no results, it falls back to returning the raw text of the document.

        Args:
            pdf_url (str): The URL of the PDF to download.
            save_dir (str, optional): The directory where the PDF will be cached.
                Defaults to "pdf_cache".
            max_pages (int, optional): The maximum number of pages to parse from the
                beginning of the PDF. Defaults to 12.
            include_sections (Optional[List[str]], optional): A list of canonical section
                names to include. Defaults to ["introduction", "methods", "limitations", "conclusion"].
            max_chars (int, optional): The maximum number of characters allowed in the
                final output string. Defaults to 16,000.

        Returns:
            str: A concatenated string of the selected sections with headers, or the
            truncated raw text if section extraction fails.
        """
    if include_sections is None:
        include_sections = ["introduction", "methods", "limitations", "conclusion"]

    pdf_path = download_pdf(pdf_url, save_dir=save_dir)
    if not pdf_path:
        return ""

    full_text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
    if not full_text:
        return ""

    sections = _find_sections_in_text(full_text)
    if not sections:
        print("[PDF] No clear sections found, falling back to raw full text snippet.")
        return full_text[:max_chars]

    selected_chunks: List[str] = []
    for sec in sections:
        name = sec["name"]
        if name in include_sections:
            chunk = full_text[sec["start"] : sec["end"]]
            header = f"\n\n===== SECTION: {name.upper()} =====\n"
            selected_chunks.append(header + chunk)

    if not selected_chunks:
        print("[PDF] No target sections found, falling back to raw full text snippet.")
        return full_text[:max_chars]

    combined = "\n".join(selected_chunks)
    if len(combined) > max_chars:
        combined = combined[:max_chars]

    return combined

