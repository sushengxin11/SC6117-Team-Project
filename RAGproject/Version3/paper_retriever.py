# paper_retriever.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import feedparser
import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# A small, conservative stopword list to avoid over-constraining queries.
# We keep it short because technical terms matter.
_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "for", "in", "on", "with",
    "via", "from", "by", "as", "at", "into", "over", "under",
}

# Common separators/hyphen variants handling
_WORD_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _to_pdf_url(entry_id: str) -> Optional[str]:
    if not entry_id:
        return None
    m = re.search(r"arxiv\.org/abs/([^?#]+)", entry_id)
    if not m:
        return None
    return f"http://arxiv.org/pdf/{m.group(1)}.pdf"


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    # unify hyphens/underscores to spaces
    s = re.sub(r"[-_]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_query(q: str) -> List[str]:
    qn = _normalize_text(q)
    parts = [p for p in _WORD_SPLIT_RE.split(qn) if p]
    tokens = [t for t in parts if t not in _STOPWORDS and len(t) >= 2]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _escape_arxiv_term(t: str) -> str:
    # arXiv API supports quoting; keep it simple and safe
    t = t.replace('"', "")
    return t


def _build_arxiv_query_strict(q: str, tokens: List[str]) -> str:
    """
    Precision-first:
    - Prefer exact phrase match in title/abstract
    - Otherwise require ALL tokens in title (ti:) or ALL in abstract (abs:)
    - Never allow partial-token matches as the final selection criterion
    """
    phrase = _escape_arxiv_term(_normalize_text(q))
    # Phrase queries: ti:"..." OR abs:"..."
    phrase_clause = f'ti:"{phrase}" OR abs:"{phrase}"'

    # ALL tokens in title
    if tokens:
        ti_all = " AND ".join([f"ti:{_escape_arxiv_term(t)}" for t in tokens])
        abs_all = " AND ".join([f"abs:{_escape_arxiv_term(t)}" for t in tokens])
        # strict query: phrase OR (all tokens in title) OR (all tokens in abstract)
        return f"({phrase_clause}) OR ({ti_all}) OR ({abs_all})"

    # Fallback: just phrase in ti/abs
    return f"({phrase_clause})"


def _build_arxiv_query_relaxed(tokens: List[str]) -> str:
    """
    Relaxed but still strong:
    - Require ALL tokens in (title OR abstract). Still no partial-token acceptance.
    """
    if not tokens:
        return "all:transformer"  # harmless fallback, should rarely happen

    # (ti:t1 OR abs:t1) AND (ti:t2 OR abs:t2) ...
    clauses = [f"(ti:{_escape_arxiv_term(t)} OR abs:{_escape_arxiv_term(t)})" for t in tokens]
    return " AND ".join(clauses)


def _extract_entry(entry) -> Dict:
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

    return {
        "title": title,
        "summary": summary,
        "link": link,
        "pdf_url": pdf_url,
        "published": published,
        "authors": authors,
    }


def _local_relevance_score(paper: Dict, phrase: str, tokens: List[str]) -> Tuple[int, Dict[str, bool]]:
    """
    Returns (score, flags).
    score is integer for stable sorting.
    flags show why it was accepted.
    """
    title_n = _normalize_text(paper.get("title", ""))
    abs_n = _normalize_text(paper.get("summary", ""))

    phrase_n = _normalize_text(phrase)
    phrase_in_title = phrase_n and (phrase_n in title_n)
    phrase_in_abs = phrase_n and (phrase_n in abs_n)

    # token containment checks
    tok_in_title = all(t in title_n for t in tokens) if tokens else False
    tok_in_abs = all(t in abs_n for t in tokens) if tokens else False
    tok_in_title_or_abs = all((t in title_n) or (t in abs_n) for t in tokens) if tokens else False

    # Base score: enforce strong signals first
    score = 0
    if phrase_in_title:
        score += 100
    if phrase_in_abs:
        score += 60

    if tok_in_title:
        score += 80
    if tok_in_abs:
        score += 40
    if tok_in_title_or_abs:
        score += 20

    # Extra: more tokens found (for longer queries)
    if tokens:
        found = sum(1 for t in tokens if (t in title_n) or (t in abs_n))
        score += found

    flags = {
        "phrase_in_title": phrase_in_title,
        "phrase_in_abs": phrase_in_abs,
        "all_tokens_in_title": tok_in_title,
        "all_tokens_in_abs": tok_in_abs,
        "all_tokens_in_title_or_abs": tok_in_title_or_abs,
    }
    return score, flags


def _dedupe_by_title(papers: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for p in papers:
        key = _normalize_text(p.get("title", ""))
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """
    Precision-first arXiv retrieval.

    Guarantees:
    - Returned papers must be strongly relevant:
        * Either exact phrase appears in title/abstract, OR
        * ALL query tokens appear in title, OR
        * ALL query tokens appear in abstract, OR
        * ALL query tokens appear across (title OR abstract)
      (No partial token matches are accepted.)
    - If not enough strong papers exist, return only those strong papers (may be < max_results).
    """
    q = (query or "").strip()
    if not q:
        return []

    tokens = _tokenize_query(q)
    # If query is too short after stopword removal, keep original words minimally
    if not tokens:
        tokens = [t for t in _tokenize_query(" ".join(_WORD_SPLIT_RE.split(_normalize_text(q)))) if t]

    # Fetch more candidates than needed, then filter locally for strong relevance
    # This is crucial because arXiv "relevance" isn't strict enough for our pipeline.
    candidate_cap = max(20, min(200, max_results * 20))

    def _fetch(search_query: str) -> List[Dict]:
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": candidate_cap,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        items = [_extract_entry(e) for e in getattr(feed, "entries", [])]
        return _dedupe_by_title(items)

    # 1) strict arXiv query
    strict_query = _build_arxiv_query_strict(q, tokens)
    candidates = _fetch(strict_query)

    # 2) local strong filtering (strict acceptance)
    phrase = q
    scored: List[Tuple[int, Dict]] = []
    for p in candidates:
        score, flags = _local_relevance_score(p, phrase, tokens)

        # Strong acceptance rule:
        # - phrase match in title/abs OR
        # - all tokens in title OR
        # - all tokens in abs OR
        # - all tokens across title/abs
        accepted = (
            flags["phrase_in_title"]
            or flags["phrase_in_abs"]
            or flags["all_tokens_in_title"]
            or flags["all_tokens_in_abs"]
            or flags["all_tokens_in_title_or_abs"]
        )
        if accepted:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    strong = [p for _, p in scored]

    # If we already have enough strong results, return top-k.
    if len(strong) >= max_results:
        return strong[:max_results]

    # 3) If insufficient, do a relaxed arXiv query but STILL enforce the same strong acceptance rule.
    # This helps when strict query misses relevant items due to arXiv parsing differences.
    relaxed_query = _build_arxiv_query_relaxed(tokens)
    more_candidates = _fetch(relaxed_query)

    # Merge, de-dupe, score again
    merged = _dedupe_by_title(strong + more_candidates)
    rescored: List[Tuple[int, Dict]] = []
    for p in merged:
        score, flags = _local_relevance_score(p, phrase, tokens)
        accepted = (
            flags["phrase_in_title"]
            or flags["phrase_in_abs"]
            or flags["all_tokens_in_title"]
            or flags["all_tokens_in_abs"]
            or flags["all_tokens_in_title_or_abs"]
        )
        if accepted:
            rescored.append((score, p))

    rescored.sort(key=lambda x: x[0], reverse=True)
    final = [p for _, p in rescored]

    # Return only truly strong matches; may be fewer than max_results (by design).
    return final[:max_results]

