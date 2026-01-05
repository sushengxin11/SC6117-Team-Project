# paper_retriever.py
#
# Functionality:
# - Fetches paper metadata (title, abstract, links, authors, published date) from arXiv API.
# - Returns a list[dict] that downstream pipeline steps can consume.

from __future__ import annotations

import re
from typing import Dict, List, Optional

import feedparser
import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _to_pdf_url(entry_id: str) -> Optional[str]:
    """
    arXiv Atom entry id typically looks like:
      http://arxiv.org/abs/YYMM.NNNNNvK
    Convert to:
      http://arxiv.org/pdf/YYMM.NNNNNvK.pdf
    """
    if not entry_id:
        return None
    m = re.search(r"arxiv\.org/abs/([^?#]+)", entry_id)
    if not m:
        return None
    return f"http://arxiv.org/pdf/{m.group(1)}.pdf"


def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """
    Fetch papers from arXiv using the official API.

    Returns items with fields:
      - title: str
      - summary: str (abstract)
      - link: str (arXiv abs page)
      - pdf_url: str | None
      - published: str | None (ISO-ish string)
      - authors: list[str]
    """
    q = (query or "").strip()
    if not q:
        return []

    # Keep the query simple and robust.
    # "all:" searches title, abstract, authors, etc.
    params = {
        "search_query": f"all:{q}",
        "start": 0,
        "max_results": int(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    papers: List[Dict] = []

    for entry in feed.entries:
        title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
        summary = (getattr(entry, "summary", "") or "").strip()
        link = getattr(entry, "link", None)
        entry_id = getattr(entry, "id", None)
        published = getattr(entry, "published", None)

        authors = []
        if hasattr(entry, "authors"):
            for a in entry.authors:
                name = getattr(a, "name", None)
                if name:
                    authors.append(str(name))

        pdf_url = None

        # Prefer explicit PDF link if present
        if hasattr(entry, "links"):
            for l in entry.links:
                href = getattr(l, "href", None)
                ltype = getattr(l, "type", None)
                rel = getattr(l, "rel", None)
                if href and ("pdf" in (ltype or "").lower() or "pdf" in (rel or "").lower() or href.endswith(".pdf")):
                    pdf_url = href
                    break

        if not pdf_url:
            pdf_url = _to_pdf_url(entry_id)

        papers.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "pdf_url": pdf_url,
                "published": published,
                "authors": authors,
            }
        )

    return papers


def debug_print_papers(papers: List[Dict], limit: int = 5) -> None:
    print(f"Fetched {len(papers)} papers.")
    for i, p in enumerate(papers[:limit]):
        print(f"== Paper {i} ==")
        print(f"  Title: {p.get('title','')}")
        print(f"  Link:  {p.get('link','')}")
        print(f"  PDF:   {p.get('pdf_url','')}")
        print("-" * 80)


if __name__ == "__main__":
    print("=== arXiv Paper Fetcher Demo ===")
    q = input("Enter a topic (e.g., 'large language models'): ").strip() or "large language models"
    papers = fetch_arxiv_papers(q, max_results=5)
    debug_print_papers(papers, limit=5)
