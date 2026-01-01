# paper_retriever.py
#
# Functionality: Extracts paper information (title + abstract + link + PDF link) from arXiv based on keywords.
# Will be used to build a RAG corpus later.

import requests
import feedparser
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """
    Fetch latest 100 papers from arXiv.
    Perform loose keyword filtering (OR match).
    Perform semantic relevance scoring using sentence-transformers.
    Return the top max_results papers that are both recent and semantically relevant.
    """

    # Step1 fetch more papers to filter from
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": 100,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=60)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
    except Exception as e:
        print(f"[Error] Failed to connect to ArXiv: {e}")
        return []

    # Step2 loose keyword OR filter & extract PDF link
    query_terms = set(query.lower().split())

    candidates = []
    for entry in feed.entries:
        title = entry.title.strip().lower()
        summary = entry.summary.strip().lower()
        text = title + " " + summary

        # OR match: as long as at least 1 keyword appears
        if not any(term in text for term in query_terms):
            continue

        pdf_url = None
        for link in entry.links:
            if link.type == 'application/pdf':
                pdf_url = link.href
                break

        # Try replacing /abs/ with /pdf/
        if not pdf_url and 'arxiv.org/abs/' in entry.link:
            pdf_url = entry.link.replace('arxiv.org/abs/', 'arxiv.org/pdf/')

        if pdf_url and not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"
        # ============================

        candidates.append({
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link,
            "pdf_url": pdf_url,
            "published": entry.published,
            "authors": [a.name for a in entry.authors],
        })

    if not candidates:
        return []

    # Step 3 semantic filtering using embeddings
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = model.encode([query], convert_to_numpy=True)
        texts = [c["title"] + " " + c["summary"] for c in candidates]
        doc_vecs = model.encode(texts, convert_to_numpy=True)

        # cosine similarity
        sims = (query_vec @ doc_vecs.T)[0]

        # sort by similarity (descending)
        ranked_idx = np.argsort(-sims)

        final_results = []
        for i in ranked_idx[:max_results]:
            final_results.append(candidates[i])

        return final_results

    except Exception as e:
        print(f"[Warning] Semantic filtering failed ({e}), returning raw results.")
        return candidates[:max_results]


def debug_print_papers(papers: List[Dict], limit: int = 5):
    """
    Simply print out a few of the scraped papers
    """
    print(f"Fetched {len(papers)} papers.")
    for i, p in enumerate(papers[:limit]):
        print(f"== Paper {i} ==")
        print(f"  Title: {p['title']}")
        print(f"  PDF:   {p.get('pdf_url', 'N/A')}")  # Print it out to see if there's a PDF.
        print("-" * 80)


if __name__ == "__main__":
    print("=== Arxiv Paper Fetcher Demo ===")
    q = input("Enter a topic (e.g., 'large language models'): ").strip()
    if not q:
        q = "large language models"

    papers = fetch_arxiv_papers(q, max_results=5)
    debug_print_papers(papers, limit=5)

