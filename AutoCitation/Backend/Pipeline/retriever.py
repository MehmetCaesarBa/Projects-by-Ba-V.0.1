import re
import time

import requests
import wikipedia
from langchain_community.retrievers import WikipediaRetriever

import Pipeline.ner as ner

# ── Wikipedia client identification ───────────────────────────────────────────
# The `wikipedia` package (used internally by LangChain's WikipediaRetriever)
# ships a generic default User-Agent that Wikimedia's bot policy now blocks
# or rate-limits — when that happens the API answers with an HTML error page,
# and the library dies with "Expecting value: line 1 column 1 (char 0)"
# (it tried to json-parse HTML). Per https://meta.wikimedia.org/wiki/User-Agent_policy
# clients must send an identifying UA with contact info.
WIKI_USER_AGENT = "AutoCitation/0.1 (PoC fact-checker; contact: mehmet17b.b@gmail.com)"

try:
    wikipedia.set_user_agent(WIKI_USER_AGENT)
except AttributeError:
    # Older package versions without the setter: patch the module global
    # the request layer reads at call time.
    wikipedia.USER_AGENT = WIKI_USER_AGENT

# Throttle successive API calls — the per-fact loop fires many requests in
# quick bursts, which is exactly the pattern that triggers rate limiting.
wikipedia.set_rate_limiting(True)

# ── Fetch retry config ────────────────────────────────────────────────────────
# One transient block/timeout should not cost the fact its entire evidence
# set (a miss cascades into a guaranteed NOT ENOUGH INFO).
RETRY_ATTEMPTS        = 3
RETRY_BACKOFF_SECONDS = 1.5

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

# OVERLAP_SENTENCES: Number of trailing sentences repeated at the start of
# the next chunk. Sentence-level overlap replaces the old 50-character
# overlap: same purpose (evidence straddling a chunk boundary is never
# lost) without ever cutting a sentence in half.
OVERLAP_SENTENCES = 1

# TOP_K_CHUNKS: Maximum chunks returned to verifier.py per atomic fact.
# AFEV paper (Section 5.5, Figure 5a) found that 1-2 evidence pieces
# per atomic fact yields optimal verification accuracy. We use 3 as a
# ceiling to give the verifier slight selection headroom.
TOP_K_CHUNKS = 2


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Wikipedia Article Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def fetch_articles(query: str) -> list[tuple[str, str]]:
    """
    Fetches Wikipedia article content for a search query using LangChain's
    WikipediaRetriever, keeping each article's canonical URL alongside its
    text (doc.metadata["source"]).

    Why keep the URL?
    The citation shown to the user must point at the article the evidence
    actually came from. Resolving a URL separately from the NER query (as
    postprocessor previously did) could cite an article the verifier never
    saw — and in practice always fell back to Special:Search.

    Args:
        query : clean search string produced by ner.extract_query()

    Returns:
        List of (article_content, article_url) tuples.
        Empty list if retrieval fails or query yields no results.
    """
    print(f"[Retriever] Fetching Wikipedia articles for query: '{query}'")

    retriever = WikipediaRetriever(
        top_k_results=TOP_K_ARTICLES,
        doc_content_chars_max=DOC_CONTENT_CHARS_MAX
    )

    docs = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            docs = retriever.invoke(query)
            break
        except Exception as e:
            # Typically a transient block/rate-limit (HTML error page →
            # JSON parse failure) or a network hiccup. Back off and retry;
            # a permanent failure must not crash the pipeline — return
            # empty and let verifier yield NOT ENOUGH INFO.
            print(f"[Retriever] Wikipedia fetch attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if docs is None:
        print(f"[Retriever] All fetch attempts failed for query '{query}'.")
        return []

    articles = [
        (doc.page_content, doc.metadata.get("source", ""))
        for doc in docs
        if doc.page_content.strip()
    ]

    print(f"[Retriever] Retrieved {len(articles)} article(s).")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Article Chunking
# ─────────────────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """
    Lightweight deterministic sentence splitter.

    Splits after sentence-final punctuation (. ! ?) when followed by
    whitespace and an uppercase letter, digit, or opening quote/paren.
    Not perfect on abbreviations, but adequate for Wikipedia prose and
    dependency-free (no spaCy inference per article).
    """
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(“])', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_articles(articles: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Splits Wikipedia articles into sentence-aware chunks of ~CHUNK_SIZE
    characters, tagging every chunk with its article URL so the final
    citation always matches the evidence actually used.

    Why sentence-aware instead of fixed character windows?
    Character windows cut mid-sentence ("...built the Statue of Liberty
    in New Yor"), destroying exactly the evidence span the verifier needs
    and rendering unreadable evidence in the UI. Packing whole sentences
    up to the size budget keeps every chunk self-contained. A chunk may
    slightly exceed CHUNK_SIZE when a single sentence is longer than the
    budget — emitted whole rather than split.

    OVERLAP_SENTENCES repeats the last sentence(s) of each chunk at the
    start of the next so boundary evidence is never lost.

    Args:
        articles : list of (article_content, article_url) tuples

    Returns:
        Flat list of (chunk, article_url) tuples across all articles.
        Empty list if no articles were provided.
    """
    if not articles:
        print("[Retriever] No articles to chunk.")
        return []

    chunks = []
    for article, url in articles:
        # Normalize whitespace so sentence boundaries are detectable.
        clean     = re.sub(r'\s+', ' ', article).strip()
        sentences = _split_sentences(clean)

        current: list[str] = []
        current_len = 0

        for sent in sentences:
            # Flush the current chunk if adding this sentence would exceed
            # the budget.
            if current and current_len + len(sent) + 1 > CHUNK_SIZE:
                chunks.append((" ".join(current), url))
                # Sentence-level overlap into the next chunk
                current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES > 0 else []
                current_len = sum(len(s) + 1 for s in current)

            current.append(sent)
            current_len += len(sent) + 1

        if current:
            chunks.append((" ".join(current), url))

    print(f"[Retriever] Generated {len(chunks)} chunk(s) across {len(articles)} article(s).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Semantic Chunk Scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_chunks(fact: str, chunks: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
    """
    Scores each chunk against the atomic fact using token-level overlap.

    Scoring formula (Jaccard-inspired coverage of the claim):
        score = |fact_tokens ∩ chunk_tokens| / |fact_tokens|

    Args:
        fact   : atomic claim string from claim_extractor
        chunks : list of (chunk, article_url) tuples from chunk_articles()

    Returns:
        List of (score, chunk, article_url) tuples sorted by descending score.
        Empty list if no chunks provided.
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
        return [(0.0, chunk, url) for chunk, url in chunks]

    scored = []
    for chunk, url in chunks:
        chunk_tokens = {
            t for t in re.findall(r'\b[a-zA-Z0-9]+\b', chunk.lower())
            if len(t) > 2
        }
        # Intersection over fact length: measures how much of the claim
        # is covered by the chunk, not how large the chunk is.
        overlap = len(fact_tokens & chunk_tokens)
        score   = overlap / len(fact_tokens)
        scored.append((score, chunk, url))

    # Sort descending: highest relevance chunks surface to the top
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"[Retriever] Top chunk score: {scored[0][0]:.2f}" if scored else "")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Top-K Chunk Selection
# ─────────────────────────────────────────────────────────────────────────────
def select_top_chunks(scored_chunks: list[tuple[float, str, str]]) -> tuple[list[str], str]:
    """
    Selects the top-k highest scoring chunks for delivery to verifier.py,
    plus the URL of the best-scoring chunk's article — used as the citation.

    Why cap at TOP_K_CHUNKS (3)?
    AFEV paper Section 5.5 Figure 5(a) demonstrates that verification
    accuracy peaks at k=1-2 evidence pieces per atomic fact. Beyond k=3,
    noise from lower-relevance chunks begins to degrade reasoning model
    performance. We use k=3 as a slight buffer to account for cases where
    the top chunk is partially relevant but not sufficient alone.

    Args:
        scored_chunks : list of (score, chunk, article_url) tuples sorted descending

    Returns:
        (list of up to TOP_K_CHUNKS chunk strings, best_chunk_article_url)
        ([], "") if no scored chunks provided.
    """
    if not scored_chunks:
        print("[Retriever] No scored chunks to select from.")
        return [], ""

    top      = [chunk for _, chunk, _ in scored_chunks[:TOP_K_CHUNKS]]
    best_url = scored_chunks[0][2]

    print(f"[Retriever] Selected {len(top)} chunk(s) for verifier.")
    return top, best_url


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# Called by main.py orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def fetch(fact: str) -> dict:
    """
    Executes the full retrieval pipeline for a single atomic fact.
    Orchestrates NER query extraction, Wikipedia article fetching,
    chunking, scoring, and top-k selection into a single callable.

    Called by claim_extractor.grounded_verify() inside the AFEV loop.

    Execution Flow:
        [Atomic Fact]
            → ner.extract_queries()        # combined + per-entity queries
            → fetch_articles(q) per query  # Wikipedia API → (text, url) pairs,
                                           #   deduplicated across queries
            → chunk_articles(articles)     # sentence-aware → url-tagged chunks
            → score_chunks(fact, chunks)   # token overlap → ranked chunks
            → select_top_chunks(scored)    # top-k filter → evidence + citation

    Args:
        fact : atomic claim string from claim_extractor

    Returns:
        {
            "chunks"     : list of up to TOP_K_CHUNKS most relevant chunks
                           (empty if retrieval failed — verifier yields
                           NOT ENOUGH INFO gracefully),
            "source_url" : URL of the article the best chunk came from
                           ("" if unavailable),
            "query"      : the NER search query used (kept for logging and
                           for postprocessor's fallback URL resolution),
        }
    """
    print(f"\n[Retriever] Starting retrieval for fact: '{fact}'")

    # Multi-query retrieval: one combined query plus one standalone query
    # per top entity (see ner.extract_queries). A single query anchored on
    # a hallucinated or mis-typed entity used to poison the whole evidence
    # pool (e.g. "Gustave Eiffel Bedford Basin" → World Heritage list).
    queries = ner.extract_queries(fact)

    articles: list[tuple[str, str]] = []
    seen = set()
    for query in queries:
        for content, url in fetch_articles(query):
            # Deduplicate articles retrieved by more than one query.
            key = url or content[:80]
            if key in seen:
                continue
            seen.add(key)
            articles.append((content, url))

    chunks             = chunk_articles(articles)
    scored_chunks      = score_chunks(fact, chunks)
    top_chunks, source = select_top_chunks(scored_chunks)

    print(
        f"[Retriever] Retrieval complete. {len(queries)} queries, "
        f"{len(articles)} unique article(s), {len(top_chunks)} chunk(s) ready for verifier."
    )
    return {"chunks": top_chunks, "source_url": source, "query": queries[0]}