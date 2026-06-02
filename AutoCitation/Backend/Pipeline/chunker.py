import re

# ── Chunk config ──────────────────────────────────────────────────────────────
# CHUNK_SIZE: Target character length per semantic chunk.
# Empirically, 400-600 characters captures one coherent topic unit from
# Wikipedia prose without splitting mid-argument or mid-sentence.
# Going below 300 risks cutting a single claim across two chunks;
# going above 700 bloats each chunk with off-topic prose that dilutes
# the token-overlap signal in the retriever's scoring step.
CHUNK_SIZE = 500

# CHUNK_OVERLAP: Shared characters between adjacent chunks.
# Overlap prevents boundary facts from being split entirely across two
# chunks, which would cause the verifier to miss evidence that straddles
# a chunk boundary. 50 characters ≈ half a short sentence — enough to
# preserve continuity without doubling the chunk count.
CHUNK_OVERLAP = 50

# MIN_CHUNK_LENGTH: Minimum character count for a chunk to be kept.
# Trailing fragments smaller than this threshold are almost always
# sentence orphans (e.g., "See also." or a dangling date) that carry
# no verification value and only add noise to the verifier's context.
MIN_CHUNK_LENGTH = 80

# SENTENCE_END_SCAN_WINDOW: How many characters to look back from the
# raw chunk boundary when searching for a clean sentence ending.
# If a natural sentence break (". ", "? ", "! ") is found within this
# window, the chunk is cut there instead of mid-word, improving
# readability and token coherence in the verifier prompt.
# Setting this too large (> CHUNK_SIZE // 2) risks under-filling chunks.
SENTENCE_END_SCAN_WINDOW = 80


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Article Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_article(article: str) -> str:
    """
    Normalizes raw Wikipedia article text before chunking.

    Wikipedia articles retrieved through LangChain's WikipediaRetriever
    often contain several categories of noise that degrade chunk quality:

    1. Section headers (e.g., "== History ==") — these are structural
       markers for the HTML render layer, not prose content. Leaving them
       in inflates token counts without adding factual signal.

    2. Reference placeholders (e.g., "[1]", "[citation needed]") —
       inline citation markers appear as tokens in the verifier's context
       and contribute nothing to the factual content of the chunk.

    3. Excessive whitespace — multi-line paragraph breaks, tab characters,
       and double spaces are collapsed to single spaces so that character-
       count-based chunking measures content density accurately.

    Args:
        article : raw Wikipedia page_content string from retriever.py

    Returns:
        Clean, single-line string with structural noise removed.
        Returns empty string if input is empty or whitespace-only.

    Example:
        Input : "== History ==\\nConstantinople[1] fell in 1453.\\n\\nMehmed II..."
        Output: "Constantinople fell in 1453. Mehmed II..."
    """
    if not article or not article.strip():
        return ""

    # Remove MediaWiki section headers: == Title ==, === Sub ==, etc.
    # These are structural delimiters, not content.
    text = re.sub(r'={2,}[^=]+={2,}', '', article)

    # Remove inline citation markers: [1], [12], [citation needed], [note 3]
    # These are render artifacts from the Wikipedia HTML → text conversion.
    text = re.sub(r'\[[^\]]{0,30}\]', '', text)

    # Collapse all whitespace variants (newlines, tabs, multiple spaces)
    # into a single space to make character-count chunking deterministic.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Sentence-Boundary Alignment
# ─────────────────────────────────────────────────────────────────────────────
def find_sentence_boundary(text: str, raw_end: int) -> int:
    """
    Finds the nearest sentence boundary at or before a raw character index.

    Pure character-window chunking without boundary alignment tends to cut
    chunks mid-word or mid-clause (e.g., "...the Ottoman Emp" / "ire fell in
    1453"). This damages token coherence in two ways:

    1. The split token ("Emp" / "ire") won't match any whole-word token in
       the verifier's prompt, dropping a potentially critical entity.
    2. The verifier's reasoning model (qwen3:8b) is instruction-tuned and
       performs significantly better on grammatically complete sentences.

    This function scans backwards from raw_end within a fixed window
    (SENTENCE_END_SCAN_WINDOW characters) looking for a terminal punctuation
    marker followed by a space (" "). If found, the chunk is cut just after
    the punctuation so each chunk ends on a complete thought.

    If no sentence boundary is found within the scan window (e.g., the text
    has an unusually long sentence), the raw character index is returned as-is
    to prevent the chunk from being under-filled beyond the threshold.

    Args:
        text    : the full cleaned article string being chunked
        raw_end : the raw character index where the chunk would end
                  under strict CHUNK_SIZE cutting

    Returns:
        Adjusted end index that aligns with a sentence boundary, or
        raw_end if no boundary was found within the scan window.

    Example:
        text    = "...Mehmed II led the siege. The city fell in 1453. Ottoman..."
        raw_end = 45  (lands mid-sentence: "The city fell in 145")
        → returns 27  (after "siege. ", cutting cleanly at sentence end)
    """
    # Define the backward-scan window: don't look further back than
    # SENTENCE_END_SCAN_WINDOW characters from the target cut point.
    scan_start = max(0, raw_end - SENTENCE_END_SCAN_WINDOW)
    window     = text[scan_start:raw_end]

    # Search for the last sentence-terminal marker within the window.
    # Patterns: ". ", "? ", "! " — space after punctuation confirms
    # the period is a sentence ender, not a decimal point or abbreviation.
    match = None
    for m in re.finditer(r'[.?!]\s', window):
        match = m  # keep iterating to find the LAST (rightmost) match

    if match:
        # Return the absolute index just after the terminal punctuation
        # (i.e., after the space that follows ".") so the next chunk
        # starts cleanly at the beginning of a new sentence.
        return scan_start + match.end()

    # No boundary found in window — return raw cut point to avoid
    # under-filling the chunk beyond what the scoring step can recover from.
    return raw_end


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Single Article Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunk_article(article: str) -> list[str]:
    """
    Splits a single preprocessed Wikipedia article into overlapping chunks.

    Chunking strategy:
    1. Preprocess the article (strip headers, refs, normalize whitespace).
    2. Slide a CHUNK_SIZE window across the text, advancing by
       (CHUNK_SIZE - CHUNK_OVERLAP) characters per step.
    3. At each cut point, align to the nearest sentence boundary within
       SENTENCE_END_SCAN_WINDOW characters using find_sentence_boundary().
    4. Discard any resulting fragment shorter than MIN_CHUNK_LENGTH to
       filter out orphaned sentence tails with no verification value.

    Why character-based sliding window over sentence tokenization?
    — spaCy sentence tokenization is slower and adds latency per article.
    — For PoC Wikipedia prose, character-window chunking with boundary
      alignment produces chunks of consistent density that are sufficient
      for token-overlap scoring in the retriever's score_chunks() step.
    — The boundary alignment step (STEP 2) recovers most mid-sentence
      cuts without needing a full sentence tokenizer.

    Args:
        article : raw Wikipedia page_content string (not yet preprocessed)

    Returns:
        List of clean, overlapping chunk strings for this article.
        Returns empty list if the article is empty after preprocessing.

    Example:
        article = "Constantinople fell in 1453. Mehmed II led the siege..."
        → [
            "Constantinople fell in 1453. Mehmed II led the siege...",
            "Mehmed II led the siege and renamed the city Istanbul...",
            ...
          ]
    """
    clean = preprocess_article(article)

    if not clean:
        print("[Chunker] Article was empty after preprocessing — skipped.")
        return []

    chunks = []
    start  = 0

    while start < len(clean):
        raw_end      = start + CHUNK_SIZE
        aligned_end  = find_sentence_boundary(clean, min(raw_end, len(clean)))
        chunk        = clean[start:aligned_end].strip()

        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
        else:
            # Fragment is too short to carry verification signal.
            # This typically occurs only on the final trailing segment.
            print(f"[Chunker] Discarding short fragment ({len(chunk)} chars): '{chunk[:40]}...'")

        # Advance start by CHUNK_SIZE minus CHUNK_OVERLAP to maintain
        # boundary continuity between adjacent chunks.
        advance = aligned_end - start - CHUNK_OVERLAP
        if advance <= 0:
            # Safety guard: if boundary alignment produced a very short
            # aligned_end, force a minimum advance to prevent an infinite loop.
            advance = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)

        start += advance

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Multi-Article Chunking
# ─────────────────────────────────────────────────────────────────────────────
def chunk_articles(articles: list[str]) -> list[str]:
    """
    Processes a list of Wikipedia articles into a flat pool of chunks.

    This is the primary interface called by retriever.py. It iterates over
    all retrieved articles, delegates per-article chunking to chunk_article(),
    and merges the results into a single flat list for scoring.

    Why merge all article chunks into one flat list?
    The scoring step in retriever.py (score_chunks) ranks chunks by token
    overlap against the atomic fact regardless of which article they came
    from. Keeping chunks from all articles in a single pool ensures the
    verifier always receives the globally most relevant evidence, not just
    the best chunk from each article in isolation.

    Args:
        articles : list of raw Wikipedia page_content strings
                   (output of retriever.fetch_articles())

    Returns:
        Flat list of all chunks across all articles.
        Returns empty list if articles is empty or all articles are blank.

    Example:
        articles = [
            "Constantinople fell in 1453...",
            "Mehmed II was the Ottoman sultan who..."
        ]
        → [
            "Constantinople fell in 1453. Mehmed II...",
            "Mehmed II directed the siege...",
            "Mehmed II was the Ottoman sultan who conquered...",
            ...
          ]
    """
    if not articles:
        print("[Chunker] No articles provided — returning empty chunk list.")
        return []

    all_chunks: list[str] = []

    for idx, article in enumerate(articles, 1):
        article_chunks = chunk_article(article)
        print(f"[Chunker] Article {idx}: generated {len(article_chunks)} chunk(s).")
        all_chunks.extend(article_chunks)

    print(f"[Chunker] Total: {len(all_chunks)} chunk(s) across {len(articles)} article(s).")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# Called by retriever.py; replaces the inline chunking logic there.
# ─────────────────────────────────────────────────────────────────────────────
def chunk(articles: list[str]) -> list[str]:
    """
    Public entry point for the chunking stage of the AutoCitation pipeline.

    This thin wrapper exists to give retriever.py a clean, single-function
    import surface (chunker.chunk(articles)) that mirrors the call style
    of the other pipeline modules (ner.extract_query, retriever.fetch, etc.).

    Internally delegates entirely to chunk_articles() for all logic.

    Integration note for retriever.py:
        Replace the inline chunk_articles() call in retriever.fetch() with:

            import chunker
            chunks = chunker.chunk(articles)

        The inline chunk_articles() and chunk_articles() functions currently
        in retriever.py should then be removed to avoid duplication.

    Args:
        articles : list of raw Wikipedia page_content strings
                   (output of retriever.fetch_articles())

    Returns:
        Flat list of all clean, overlapping chunks across all articles,
        ready for scoring in retriever.score_chunks().

    Example:
        fetch("Constantinople was conquered by Turks in 1453")
        articles → chunker.chunk(articles)
        → [
            "Constantinople fell to Ottoman forces on May 29, 1453...",
            "Mehmed II led the siege that ended the Byzantine Empire...",
            ...
          ]
    """
    return chunk_articles(articles)