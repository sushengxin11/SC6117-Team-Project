import fitz
import re
import json
from typing import List, Dict, Any
from llm_client import call_llm_raw
from pathlib import Path
from pdf_utils import download_pdf
import concurrent.futures


# Remove References section
def strip_references(text: str) -> str:
    """
    Intelligently removes references but retains subsequent appendices/supplementary materials.
    Supports recognition of: Appendix, Supplementary Material, Supplemental Data, and other variations.
    """
    # 1. Find the starting point for reference
    ref_pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:REFERENCES|BIBLIOGRAPHY)\s*(?:\n|$)",
        re.IGNORECASE
    )

    ref_match = ref_pattern.search(text)

    # If "References" cannot be found throughout the entire document, return
    if not ref_match:
        print("[SmartClean] No 'References' section found. Keeping full text.")
        return text

    ref_start = ref_match.start()
    ref_end = ref_match.end()

    # 2. Find the starting point for appendices/supplementary materials
    text_after_ref = text[ref_end:]

    app_pattern = re.compile(
        r"(?:^|\n)\s*(?:[A-Z0-9]+\.?\s*)?"
        r"(?:APPENDIX|APPENDICES|SUPPLEMENTARY|SUPPLEMENTAL|DATA AVAILABILITY)"
        r"(?:\s+(?:MATERIAL|DATA|INFORMATION|section))?"
        r"\b",
        re.IGNORECASE
    )

    app_match = app_pattern.search(text_after_ref)

    if app_match:

        app_start_absolute = ref_end + app_match.start()

        print(f"[SmartClean] Strategy: HOLLOW OUT. Removing text between {ref_start} and {app_start_absolute}.")

        cleaned_text = (
                text[:ref_start] +
                "\n\n--- [META: References list removed to save tokens] ---\n\n" +
                text[app_start_absolute:]
        )
        return cleaned_text

    else:

        print(f"[SmartClean] Strategy: TRUNCATE. Removed everything after char {ref_start}.")
        return text[:ref_start]


# 2. Auto-detect optimal chunk size
#    Based on:
#    - average paragraph length
#    - number of lines
#    - formula density

def auto_detect_chunk_size(text: str) -> int:
    paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 0]

    avg_len = sum(len(p) for p in paragraphs) / max(1, len(paragraphs))
    formula_count = len(re.findall(r"(\$.*?\$|\\\[|\\\]|\\begin{equation}|\\end{equation})", text))

    # base size
    chunk_size = 4000

    # increase for long paragraphs
    if avg_len > 600:
        chunk_size += 2000
    if avg_len > 1200:
        chunk_size += 2000

    # increase if many formulas
    if formula_count > 10:
        chunk_size += 2000
    if formula_count > 25:
        chunk_size += 2000

    # clamp
    chunk_size = max(3000, min(chunk_size, 9000))
    print(f"[SmartChunk] Auto-detected chunk size = {chunk_size}")

    return chunk_size


# 3. Detect PDF sections (Improved: Strict Headers Only)


RAW_TITLES = [
    r"introduction",
    r"related work|background",
    r"method|methods|methodology|approach",
    r"model|architecture",
    r"experiments?|results?|evaluation",
    r"limitations?",
    r"discussion",
    r"conclusions?|future work"
]

# Building Stricter Regular Expressions

SECTION_PATTERN = (
        r"(?:^|\n)\s*"
        r"(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?"
        r"\b(" + "|".join(RAW_TITLES) + r")\b"
)

SECTION_REGEX = re.compile(SECTION_PATTERN, re.IGNORECASE)


def split_by_sections(text: str) -> List[str]:
    matches = list(SECTION_REGEX.finditer(text))

    if not matches:
        return [text]

    sections = []

    current_start = 0

    for i, m in enumerate(matches):

        match_start = m.start()

        if match_start > current_start:

            if match_start - current_start > 50:
                sections.append(text[current_start:match_start])
                current_start = match_start
            else:

                pass

    if current_start < len(text):
        sections.append(text[current_start:])

    print(f"[SmartChunk] Detected {len(sections)} high-confidence sections.")
    return sections


# 4. Ensure formulas are not split: auto merge chunks
FORMULA_OPEN = [r"\$", r"\\\["]
FORMULA_CLOSE = [r"\$", r"\\\]"]


def is_formula_incomplete(chunk: str) -> bool:
    """
    Determine if a chunk ends inside a formula block.
    """
    # Count occurrences of various math delimiters
    opens = len(re.findall(r"\$", chunk))
    if opens % 2 == 1:
        return True  # odd number of $ means incomplete inline math

    # Block math check
    block_open = len(re.findall(r"\\\[", chunk))
    block_close = len(re.findall(r"\\\]", chunk))
    if block_open != block_close:
        return True

    return False


def merge_formula_chunks(chunks: List[str]) -> List[str]:
    merged = []
    buffer = ""

    for chunk in chunks:
        if not buffer:
            buffer = chunk
        else:
            buffer += chunk

        # Check if buffer ends inside a formula
        if not is_formula_incomplete(buffer):
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    print(f"[SmartChunk] After formula-merge: {len(merged)} chunks.")
    return merged


# 5. Main smart chunking function (full upgrade)
def smart_chunk_pdf(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)

    # Extract all text
    full_text = "\n".join([page.get_text("text") for page in doc])

    # Remove references
    full_text = strip_references(full_text)

    # Auto-detect optimal chunk size
    optimal_size = auto_detect_chunk_size(full_text)

    # First split by high-level sections
    sections = split_by_sections(full_text)

    # Second split sections by optimal chunk size
    raw_chunks = []
    for sec in sections:
        if len(sec) <= optimal_size:
            raw_chunks.append(sec)
        else:
            # further split inside section
            for start in range(0, len(sec), optimal_size):
                raw_chunks.append(sec[start:start + optimal_size])

    print(f"[SmartChunk] Initial chunks before formula merge: {len(raw_chunks)}")

    # Third: merge chunks that break formulas
    final_chunks = merge_formula_chunks(raw_chunks)

    print(f"[SmartChunk] Final chunk count = {len(final_chunks)}")
    return final_chunks


# 6. Use the new smart chunking in main pipeline
CHUNK_SUMMARY_SYSTEM_PROMPT = """
You are an expert ML researcher.
You receive *part of a scientific PDF* (a chunk).
Your job is NOT to summarize the chunk, but to EXTRACT STRUCTURED TECHNICAL CONTENT.

You MUST output STRICT JSON with fields:
{
  "math": [],
  "methods": [],
  "experiments": [],
  "limitations": []
}
"""


def summarize_chunk(chunk_text: str) -> Dict[str, Any]:
    user_prompt = f"""
Extract structured technical content.

----- BEGIN CHUNK -----
{chunk_text}
----- END CHUNK -----
"""
    raw = call_llm_raw(
        system_prompt=CHUNK_SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=1500
    )

    import json
    try:
        return json.loads(raw)
    except:
        return {"math": [], "methods": [], "experiments": [], "limitations": []}


def aggregate_chunk_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge the structured results of each chunk.
    """
    merged: Dict[str, List[Any]] = {k: [] for k in ["math", "methods", "experiments", "limitations"]}

    # 1. simple assembly
    for s in summaries:
        for k in merged:
            if k in s and isinstance(s[k], list):
                merged[k].extend(s[k])

    # 2. Perform "stringification + deduplication" uniformly.
    for k in merged:
        normalized_items: List[str] = []

        for x in merged[k]:
            if isinstance(x, str):
                s_val = x.strip()
            # Serializing dict/list using JSON
            elif isinstance(x, (dict, list)):
                try:
                    s_val = json.dumps(x, ensure_ascii=False, sort_keys=True).strip()
                except Exception:
                    s_val = str(x).strip()
            else:
                s_val = str(x).strip()

            if s_val:
                normalized_items.append(s_val)

        merged[k] = list(dict.fromkeys(normalized_items))

    return merged


def process_pdf_fullscan(
        pdf_url: str,
        max_workers: int = 6,
        cache_dir: str = "pdf_structured_cache",
) -> Dict[str, Any]:
    """
    The entire PDF is intelligently segmented and structured for extraction, and the results are cached locally as JSON.
    The next time the same PDF is accessed, it is read from the cache first, avoiding repeated full scans.
    """
    pdf_path = download_pdf(pdf_url)
    if not pdf_path:
        return {}

    # 1. Prepare cache path
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(pdf_path).stem  # e.g. 1234.5678v1
    cache_file = cache_dir_path / f"{pdf_stem}.json"

    # 2. If the cache already exists, load it directly.
    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"[PDF FullScan] Loaded structured summary from cache: {cache_file}")
            return cached
        except Exception as e:
            print(f"[PDF FullScan] Failed to load cache {cache_file}, will recompute. Error: {e}")

    # 3. No cache, Normal fullscan process
    chunks = smart_chunk_pdf(pdf_path)
    total = len(chunks)
    print(f"[PDF FullScan] Total chunks: {total}")

    summaries: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(summarize_chunk, chunk) for chunk in chunks
        ]

        for i, f in enumerate(futures):
            res = f.result()
            print(f"[PDF Processor] Completed chunk {i + 1}/{total}")
            summaries.append(res)

    aggregated = aggregate_chunk_summaries(summaries)

    # 4. Write to cache
    try:
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        print(f"[PDF FullScan] Saved structured summary to cache: {cache_file}")
    except Exception as e:
        print(f"[PDF FullScan] Failed to save cache {cache_file}: {e}")

    return aggregated