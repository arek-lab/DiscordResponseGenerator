"""
Tavily crawl data cleaning pipeline for Lovable docs.
Reads .pkl files with LangChain Documents, strips site boilerplate,
cleans noise, deduplicates, and saves to a new folder.
"""

import re
import pickle
import logging
from pathlib import Path
from copy import deepcopy

from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Strip Lovable site boilerplate (prefix + suffix)
#
# Every page from docs.lovable.dev starts with identical navigation HTML
# rendered as Markdown, and ends with a feedback footer.
# We extract ONLY the content between the first real heading and the footer.
# ---------------------------------------------------------------------------

# The nav block always ends just before the first H1/H2/H3 heading of real content.
_FIRST_HEADING = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)

# Footer markers — everything from these strings onwards is noise
_FOOTER_MARKERS = [
    "Was this page helpful?",
    "PreviousNext",
    "[Previous]",
    "Edit this page",
    "Last updated on",
    "© 2024 Lovable",
    "© 2025 Lovable",
]


def _strip_lovable_boilerplate(text: str) -> str:
    """
    Remove the shared navigation prefix and footer suffix present on
    every docs.lovable.dev page captured by Tavily.

    Before:
        [Lovable Documentation home page](...)
        [Introduction](...)[Features](...)...
        * [Feb 5, 2026](#feb-5-2026)   ← sidebar TOC (different per page!)
        ...
        # Real page heading
        Actual content here...
        Was this page helpful?

    After:
        # Real page heading
        Actual content here...
    """
    # Strip prefix: everything before the first real heading
    match = _FIRST_HEADING.search(text)
    if match:
        text = text[match.start():]
    else:
        # No heading at all — page is pure nav/boilerplate
        return ""

    # Strip suffix: everything from footer markers onwards
    for marker in _FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    return text.strip()


# ---------------------------------------------------------------------------
# Step 2 — Clean inline noise from the body content
# ---------------------------------------------------------------------------

_INLINE_PATTERNS = [
    (re.compile(r"!\[.*?\]\(.*?\)"),         ""),      # markdown images
    (re.compile(r"\[([^\]]+)\]\([^\)]+\)"),  r"\1"),   # links → keep anchor text
    (re.compile(r"<[^>]{1,100}>"),           ""),      # residual HTML tags
    (re.compile(r"https?://\S+"),            ""),      # bare URLs
    (re.compile(r"={3,}|-{3,}|\*{3,}"),     ""),      # horizontal rules
    (re.compile(r"^\s*[\*\-]\s*$", re.M),   ""),      # lone bullet markers
    (re.compile(r"\[\s*\]\([^\)]*\)"),       ""),      # empty link text []()
    (re.compile(r"<!--.*?-->", re.DOTALL),  ""),      # HTML comments
    # Lovable-specific: invisible anchor links [​](#some-section)
    (re.compile(r"\[​?\]\(#[^\)]*\)"),       ""),
]

# Line-level patterns — catches anything that slipped past the prefix strip
_NAV_LINE = re.compile(
    r"""
    ^(Home|Docs?|API|Guide|Reference|Examples?|Tutorial|Changelog)\s*[>›»\|]
    | ^Skip\s+to\s+
    | ^(Last\s+updated|Edit\s+this\s+page|Report\s+an\s+issue)\s*[:\-]?
    | ^\d+\s+min\s+read
    | ^Tags?:\s
    | ^(Search|GitHub|Discord|Twitter|LinkedIn|YouTube)\s*$
    | ^(Copyright|©)\s*\d{4}
    | ^All\s+rights\s+reserved
    | ^Cookie\s+(Policy|Settings|Preferences)
    | ^Accept\s+(all\s+)?[Cc]ookies
    | ^(Previous|Next)\s*[:\-]?\s*(page|article|section)
    | ^Table\s+of\s+[Cc]ontents\s*$
    | ^On\s+this\s+page\s*$
    | ^In\s+this\s+(article|section|guide)\s*$
    # Lovable sidebar date entries still present on some pages
    | ^\*\s+\[(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+,?\s+\d{4}\]
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
_EXCESSIVE_SPACES   = re.compile(r"[ \t]{2,}")
_SHORT_LINE         = re.compile(r"^.{1,15}$")


def _clean_body(text: str) -> str:
    """Clean noise from the body content (after boilerplate stripping)."""

    # 1. Inline substitutions
    for pattern, replacement in _INLINE_PATTERNS:
        text = pattern.sub(replacement, text)

    # 2. Remove residual nav / boilerplate lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if _NAV_LINE.match(stripped):
            continue
        # Drop very short lines that aren't structural markdown
        if _SHORT_LINE.match(stripped) and not stripped.startswith(("#", "`", ">", "-", "*")):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 3. Normalise whitespace
    text = _EXCESSIVE_NEWLINES.sub("\n\n", text)
    text = _EXCESSIVE_SPACES.sub(" ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _dedup_by_url(
    docs: list[Document],
    global_seen_urls: set[str],
    keep: str = "longest",
) -> tuple[list[Document], int]:
    """
    Deduplicate by source URL across all files.

    For each URL keeps either the 'longest' or 'first' seen document.
    'longest' is better for Lovable docs because some crawl runs returned
    truncated versions of the same page.

    Parameters
    ----------
    docs             : cleaned documents for current file
    global_seen_urls : shared set across all files (mutated in place)
    keep             : 'longest' | 'first'

    Returns
    -------
    (unique_docs, n_removed)
    """
    local_best: dict[str, Document] = {}

    for doc in docs:
        url = doc.metadata.get("source", "")

        if url in global_seen_urls:
            continue  # already have this URL from a previous file

        if keep == "longest":
            if url not in local_best or len(doc.page_content) > len(local_best[url].page_content):
                local_best[url] = doc
        else:
            if url not in local_best:
                local_best[url] = doc

    unique = list(local_best.values())
    global_seen_urls.update(local_best.keys())

    removed = len(docs) - len(unique)
    return unique, removed


# ---------------------------------------------------------------------------
# Step 3 — Combine into single clean_document()
# ---------------------------------------------------------------------------

def clean_document(doc: Document, min_length: int = 50) -> Document | None:
    """
    Full cleaning pipeline for a single LangChain Document:
      1. Strip Lovable site boilerplate (nav prefix + footer suffix)
      2. Clean inline noise from body content
      3. Return None if result is too short to be useful

    Parameters
    ----------
    doc        : source Document with raw Tavily content
    min_length : minimum character count after cleaning; shorter docs are dropped
    """
    text = doc.page_content

    # Stage 1: strip site chrome (the main fix for 95% dedup problem)
    text = _strip_lovable_boilerplate(text)
    if not text:
        return None

    # Stage 2: clean body noise
    text = _clean_body(text)
    if len(text) < min_length:
        return None

    new_doc = deepcopy(doc)
    new_doc.page_content = text
    return new_doc


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_docs_folder(
    source_folder: str | Path,
    output_folder: str | Path | None = None,
    deduplicate: bool = True,
    min_doc_length: int = 50,
) -> dict:
    """
    Read all .pkl files from *source_folder*, clean the contained Documents,
    deduplicate across all files, and write cleaned files to *output_folder*.

    Parameters
    ----------
    source_folder  : folder with raw .pkl files
    output_folder  : destination; defaults to <source_parent>/cleaned_docs
    deduplicate    : drop exact-duplicate documents across all files
    min_doc_length : minimum character length to keep a document

    Returns
    -------
    dict with processing stats
    """
    source_path = Path(source_folder).resolve()
    output_path = Path(output_folder).resolve() if output_folder else source_path.parent / "cleaned_docs"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Source : %s", source_path)
    logger.info("Output : %s", output_path)

    pkl_files = sorted(source_path.glob("*.pkl"))
    if not pkl_files:
        logger.warning("No .pkl files found in %s", source_path)
        return {}

    total_before      = 0
    total_after       = 0
    total_rm_cleaning = 0
    total_rm_dedup    = 0
    files_ok          = 0

    global_seen_urls: set[str] = set()  # dedup by source URL across all files

    for filepath in pkl_files:
        logger.info("Processing: %s", filepath.name)

        try:
            with open(filepath, "rb") as f:
                docs: list[Document] = pickle.load(f)
        except Exception as exc:
            logger.error("  Failed to load %s: %s", filepath.name, exc)
            continue

        n_before = len(docs)
        total_before += n_before

        # --- Clean each document (boilerplate strip + body noise) ---
        cleaned: list[Document] = []
        for doc in docs:
            result = clean_document(doc, min_length=min_doc_length)
            if result is not None:
                cleaned.append(result)

        removed_by_cleaning = n_before - len(cleaned)

        # --- Deduplicate by source URL across all files ---
        removed_by_dedup = 0
        if deduplicate:
            cleaned, removed_by_dedup = _dedup_by_url(cleaned, global_seen_urls, keep="longest")

        total_after       += len(cleaned)
        total_rm_cleaning += removed_by_cleaning
        total_rm_dedup    += removed_by_dedup

        logger.info(
            "  %d → %d docs  (-%d boilerplate/noise, -%d dedup)",
            n_before, len(cleaned), removed_by_cleaning, removed_by_dedup,
        )

        out_path = output_path / filepath.name
        with open(out_path, "wb") as f:
            pickle.dump(cleaned, f)

        files_ok += 1

    stats = {
        "files_processed"  : files_ok,
        "docs_before"      : total_before,
        "docs_after"       : total_after,
        "removed_cleaning" : total_rm_cleaning,
        "removed_dedup"    : total_rm_dedup,
        "reduction_pct"    : round(100 * (total_before - total_after) / max(total_before, 1), 1),
    }

    logger.info("=" * 55)
    logger.info(
        "Done. Files: %d | Docs: %d → %d | "
        "-cleaning: %d | -dedup: %d | total reduction: %.1f%%",
        stats["files_processed"],
        stats["docs_before"],
        stats["docs_after"],
        stats["removed_cleaning"],
        stats["removed_dedup"],
        stats["reduction_pct"],
    )
    logger.info("Saved to: %s", output_path)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean Tavily crawl .pkl files.")
    parser.add_argument("source",  help="Folder with raw .pkl files")
    parser.add_argument("--output", default=None, help="Output folder (default: ../cleaned_docs)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip cross-file deduplication")
    parser.add_argument("--min-length", type=int, default=50, help="Min doc length after cleaning")
    args = parser.parse_args()

    stats = clean_docs_folder(
        source_folder=args.source,
        output_folder=args.output,
        deduplicate=not args.no_dedup,
        min_doc_length=args.min_length,
    )
    print(stats)