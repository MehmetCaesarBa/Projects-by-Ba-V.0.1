import re
import requests
from langchain_community.retrievers import WikipediaRetriever

import Pipeline.ner as ner

# ── Wikipedia Retriever config ────────────────────────────────────────────────
# top_k_results: Number of Wikipedia articles fetched per query.
# Two articles provide enough surface area to cover multi-hop claims
# without flooding the chunker with irrelevant documents.
TOP_K_ARTICLES = 2

# doc_content_chars_max: Hard character ceiling per fetched Wikipedia article.
# Wikipedia articles can exceed 100k characters. Uncapped retrieval would
# overflow the reasoning model's context window on 7-8B parameter local models.
# 4000 characters ≈ ~800 tokens, a safe ceiling for local inference.
DOC_CONTENT_CHARS_MAX = 4000

# ── Chunk config ──────────────────────────────────────────────────────────────
# CHUNK_SIZE: Target character length per semantic chunk.
# Empirically, 400-600 characters captures one coherent topic unit from
# Wikipedia prose without splitting mid-argument or mid-sentence.
CHUNK_SIZE = 500

# CHUNK_OVERLAP: Shared characters between adjacent chunks.
# Overlap prevents boundary facts from being split across two chunks,
# which would cause the verifier to miss evidence straddling a boundary.
CHUNK_OVERLAP = 50

# TOP_K_CHUNKS: Maximum chunks returned to verifier.py per atomic fact.
# AFEV paper (Section 5.5, Figure 5a) found that 1-2 evidence pieces
# per atomic fact yields optimal verification accuracy. We use 3 as a
# ceiling to give the verifier slight selection headroom.
TOP_K_CHUNKS = 3


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Wikipedia Article Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def fetch_articles(query: str) -> list[str]:
    """
    Fetches raw Wikipedia article content for a given search query
    using LangChain's WikipediaRetriever.

    WikipediaRetriever internally uses the Wikipedia API's search endpoint,
    returning the top-k most relevant articles as LangChain Document objects.
    We extract only the page_content field from each document.

    Why LangChain's retriever over raw Wikipedia API calls?
    LangChain handles search disambiguation, redirect resolution, and
    content extraction in one call, removing the need for a custom
    Wikipedia parsing layer at PoC stage.

    Args:
        query : clean search string produced by ner.extract_query()

    Returns:
        List of raw article content strings, one per article.
        Returns empty list if retrieval fails or query yields no results.

    Example:
        fetch_articles("Fall of Constantinople")
        → ["Constantinople (Turkish: İstanbul) fell on May 29, 1453...",
           "Mehmed II was the Ottoman sultan who conquered..."]
    """
    print(f"[Retriever] Fetching Wikipedia articles for query: '{query}'")

    retriever = WikipediaRetriever(
        top_k_results=TOP_K_ARTICLES,
        doc_content_chars_max=DOC_CONTENT_CHARS_MAX
    )

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        # Network failure or Wikipedia API unavailability should not crash
        # the pipeline — return empty and let verifier handle missing evidence.
        print(f"[Retriever] Wikipedia fetch failed: {e}")
        return []

    articles = [doc.page_content for doc in docs if doc.page_content.strip()]

    print(f"[Retriever] Retrieved {len(articles)} article(s).")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Article Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunk_articles(articles: list[str]) -> list[str]:
    """
    Splits raw Wikipedia articles into fixed-size overlapping character chunks.

    Why chunk instead of passing full articles to the verifier?
    Local 7-8B reasoning models have limited context windows (~4096-8192 tokens).
    Passing a full Wikipedia article risks truncation of the exact evidence
    the model needs. Chunking ensures the verifier receives dense, focused
    segments rather than diluted full-article context.

    Why character-based chunking instead of sentence-based?
    Sentence tokenization adds a spaCy dependency and is slower.
    Character-based chunking with overlap is fast, deterministic, and
    sufficient for PoC evidence segmentation since chunks still respect
    approximate sentence boundaries at 500-character windows.

    Overlap strategy:
    A 50-character overlap between adjacent chunks prevents a critical
    boundary fact (e.g., a date or name appearing at the end of one chunk
    and the start of the next) from being split and missed entirely.

    Args:
        articles : list of raw article content strings from fetch_articles()

    Returns:
        Flat list of all chunks across all articles.
        Empty list if no articles were provided.

    Example:
        chunk_articles(["Constantinople fell in 1453. The city..."])
        → ["Constantinople fell in 1453. The city was...",
           "The city was renamed Istanbul after..."]
    """
    if not articles:
        print("[Retriever] No articles to chunk.")
        return []

    chunks = []
    for article in articles:
        # Normalize whitespace before chunking to prevent
        # large blank gaps from inflating chunk boundaries artificially.
        clean = re.sub(r'\s+', ' ', article).strip()

        start = 0
        while start < len(clean):
            end   = start + CHUNK_SIZE
            chunk = clean[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Advance by CHUNK_SIZE minus OVERLAP to maintain boundary continuity
            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"[Retriever] Generated {len(chunks)} chunk(s) across {len(articles)} article(s).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Semantic Chunk Scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_chunks(fact: str, chunks: list[str]) -> list[tuple[float, str]]:
    """
    Scores each chunk against the atomic fact using token-level overlap.

    Why token overlap instead of sentence-transformers embeddings?
    Sentence-transformers (e.g., all-MiniLM-L6-v2) would give more
    semantically precise scoring, but requires a model load (~80MB)
    and GPU/CPU inference per chunk. For the PoC, token overlap is a
    fast, zero-VRAM approximation that performs adequately on factual
    claims containing named entities and dates — exactly the content
    that atomic facts are built from.

    Token overlap scoring logic:
    Jaccard-inspired intersection: count how many unique tokens from
    the fact appear in the chunk, normalized by the fact's token count.
    This gives a 0.0–1.0 relevance score per chunk.

    Scoring formula:
        score = |fact_tokens ∩ chunk_tokens| / |fact_tokens|

    Args:
        fact   : atomic claim string from claim_extractor
        chunks : flat list of all chunks from chunk_articles()

    Returns:
        List of (score, chunk) tuples sorted by descending score.
        Empty list if no chunks provided.

    Example:
        fact = "Constantinople fell in 1453"
        chunk = "The city of Constantinople fell to Ottoman forces in 1453..."
        → score ≈ 0.75  (3 of 4 content tokens matched)
    """
    if not chunks:
        print("[Retriever] No chunks available to score.")
        return []

    # Normalize fact to lowercase token set for case-insensitive matching.
    # Short tokens (≤2 chars) are excluded as they are typically articles
    # or prepositions that add noise without semantic value.
    fact_tokens = {
        t for t in re.findall(r'\b[a-zA-Z0-9]+\b', fact.lower())
        if len(t) > 2
    }

    if not fact_tokens:
        # If the fact contains no meaningful tokens after filtering,
        # return all chunks unscored with equal weight of 0.0
        print("[Retriever] No scoreable tokens found in fact. Returning unscored chunks.")
        return [(0.0, chunk) for chunk in chunks]

    scored = []
    for chunk in chunks:
        chunk_tokens = {
            t for t in re.findall(r'\b[a-zA-Z0-9]+\b', chunk.lower())
            if len(t) > 2
        }
        # Intersection over fact length: measures how much of the claim
        # is covered by the chunk, not how large the chunk is.
        overlap = len(fact_tokens & chunk_tokens)
        score   = overlap / len(fact_tokens)
        scored.append((score, chunk))

    # Sort descending: highest relevance chunks surface to the top
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"[Retriever] Top chunk score: {scored[0][0]:.2f}" if scored else "")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Top-K Chunk Selection
# ─────────────────────────────────────────────────────────────────────────────
def select_top_chunks(scored_chunks: list[tuple[float, str]]) -> list[str]:
    """
    Selects the top-k highest scoring chunks for delivery to verifier.py.

    Why cap at TOP_K_CHUNKS (3)?
    AFEV paper Section 5.5 Figure 5(a) demonstrates that verification
    accuracy peaks at k=1-2 evidence pieces per atomic fact. Beyond k=3,
    noise from lower-relevance chunks begins to degrade reasoning model
    performance. We use k=3 as a slight buffer to account for cases where
    the top chunk is partially relevant but not sufficient alone.

    Args:
        scored_chunks : list of (score, chunk) tuples sorted descending

    Returns:
        List of up to TOP_K_CHUNKS chunk strings, highest relevance first.
        Returns empty list if no scored chunks provided.

    Example:
        scored_chunks = [(0.85, "chunk A"), (0.60, "chunk B"), (0.20, "chunk C")]
        → ["chunk A", "chunk B", "chunk C"]  (all 3 within TOP_K_CHUNKS=3)
    """
    if not scored_chunks:
        print("[Retriever] No scored chunks to select from.")
        return []

    top = [chunk for _, chunk in scored_chunks[:TOP_K_CHUNKS]]

    print(f"[Retriever] Selected {len(top)} chunk(s) for verifier.")
    return top


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# Called by main.py orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def fetch(fact: str) -> list[str]:
    """
    Executes the full retrieval pipeline for a single atomic fact.
    Orchestrates NER query extraction, Wikipedia article fetching,
    chunking, scoring, and top-k selection into a single callable.

    This is the only function main.py needs to call from this module.
    verifier.py receives the output directly as its evidence_chunks argument.

    Execution Flow:
        [Atomic Fact]
            → ner.extract_query()          # GPE/PERSON/EVENT → search string
            → fetch_articles(query)        # Wikipedia API → raw article text
            → chunk_articles(articles)     # sliding window → chunk list
            → score_chunks(fact, chunks)   # token overlap → ranked chunks
            → select_top_chunks(scored)    # top-k filter → evidence list
            → [Evidence Chunks] → verifier.verify(fact, evidence_chunks)

    Args:
        fact : atomic claim string from claim_extractor

    Returns:
        List of up to TOP_K_CHUNKS most relevant text chunks.
        Returns empty list if Wikipedia retrieval fails entirely,
        allowing verifier to output NOT ENOUGH INFO gracefully.

    Example:
        fetch("Constantinople was conquered by Turks in 1453")
        → [
            "Constantinople fell to Ottoman forces on May 29, 1453...",
            "Mehmed II led the siege that ended the Byzantine Empire...",
            "The conquest marked the end of the Middle Ages in Eastern Europe..."
          ]
    """
    print(f"\n[Retriever] Starting retrieval for fact: '{fact}'")

    query         = ner.extract_query(fact)
    articles      = fetch_articles(query)
    chunks        = chunk_articles(articles)
    scored_chunks = score_chunks(fact, chunks)
    top_chunks    = select_top_chunks(scored_chunks)

    print(f"[Retriever] Retrieval complete. {len(top_chunks)} chunk(s) ready for verifier.")
    return top_chunks