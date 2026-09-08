"""
Unit tests for the chunk ranker in Pipeline/retriever.py.

Pure scoring — no network, no Wikipedia, no model. The chunks below are
shortened versions of passages this pipeline actually retrieved, so the tests
reproduce a real ranking failure rather than an invented one.
"""

import pytest

from Pipeline import retriever


# The failure that motivated IDF weighting. A claim about a strait retrieved a
# passage about African settlement patterns, which scored 0.42 under unweighted
# term coverage because it matched 'africa', 'europe' and 'between' — three
# common words — while missing 'bosporus' entirely.
CLAIM = "The Bosporus separates Africa and Europe."

RELEVANT = (
    "The Bosporus is a natural strait in Turkey. The Bosporus connects the "
    "Black Sea to the Sea of Marmara and forms one of the continental "
    "boundaries between Asia and Europe.",
    "https://en.wikipedia.org/wiki/Bosporus",
)

DISTRACTOR = (
    "A modest annual rainfall was considered suitable for temperate farming. "
    "The first parts of Africa to be populated by Europeans were located at "
    "the northern and southern extremities, and between these two extremes "
    "disease precluded most permanent European settlement.",
    "https://en.wikipedia.org/wiki/White_Africans",
)

# Filler chunks make 'africa' and 'europe' common across the retrieved set,
# which is exactly the situation IDF is meant to exploit. Without them there is
# no document frequency signal to compute.
FILLER = [
    (f"Europe and Africa have been connected by trade routes for centuries, "
     f"and travel between Europe and Africa was common in period {i}.", "u")
    for i in range(6)
]

ALL_CHUNKS = [RELEVANT, DISTRACTOR, *FILLER]


def _rank_of(scored, chunk_text):
    """Position of a chunk in the sorted results, 0 = first."""
    return next(i for i, (_, c, _) in enumerate(scored) if c == chunk_text)


# ═════════════════════════════════════════════════════════════════════════════
# The behaviour IDF was added for
# ═════════════════════════════════════════════════════════════════════════════
def test_relevant_chunk_outranks_the_common_word_distractor():
    """
    'bosporus' appears in one chunk; 'africa' and 'europe' appear in most.
    Weighting by rarity must put the chunk containing the rare term first.
    """
    scored = retriever.score_chunks(CLAIM, ALL_CHUNKS)
    assert _rank_of(scored, RELEVANT[0]) == 0


def test_distractor_scores_below_the_relevant_chunk():
    scored = retriever.score_chunks(CLAIM, ALL_CHUNKS)
    by_text = {c: s for s, c, _ in scored}
    assert by_text[RELEVANT[0]] > by_text[DISTRACTOR[0]]


def test_idf_widens_the_gap_versus_plain_coverage(monkeypatch):
    """
    The point of the change, stated as a measurement: switching weighting off
    should shrink the margin between the right chunk and the distractor.
    """
    scored_idf = retriever.score_chunks(CLAIM, ALL_CHUNKS)
    idf_gap = ({c: s for s, c, _ in scored_idf}[RELEVANT[0]]
               - {c: s for s, c, _ in scored_idf}[DISTRACTOR[0]])

    monkeypatch.setattr(retriever, "USE_IDF_WEIGHTING", False)
    scored_plain = retriever.score_chunks(CLAIM, ALL_CHUNKS)
    plain_gap = ({c: s for s, c, _ in scored_plain}[RELEVANT[0]]
                 - {c: s for s, c, _ in scored_plain}[DISTRACTOR[0]])

    assert idf_gap > plain_gap


# ═════════════════════════════════════════════════════════════════════════════
# Invariants that must survive any future ranker change
# ═════════════════════════════════════════════════════════════════════════════
def test_scores_stay_within_zero_and_one():
    """
    select_top_chunks and the logs both treat the score as a 0-1 coverage
    figure; an unbounded score would silently break their meaning.
    """
    for score, _, _ in retriever.score_chunks(CLAIM, ALL_CHUNKS):
        assert 0.0 <= score <= 1.0


def test_results_are_sorted_descending():
    scores = [s for s, _, _ in retriever.score_chunks(CLAIM, ALL_CHUNKS)]
    assert scores == sorted(scores, reverse=True)


def test_every_chunk_is_returned():
    """Ranking reorders; TOP_K_CHUNKS does the filtering, further down."""
    assert len(retriever.score_chunks(CLAIM, ALL_CHUNKS)) == len(ALL_CHUNKS)


def test_urls_stay_attached_to_their_chunks():
    """A shuffled url would cite the wrong article for the right evidence."""
    for _, chunk, url in retriever.score_chunks(CLAIM, ALL_CHUNKS):
        assert dict((c, u) for c, u in ALL_CHUNKS)[chunk] == url


# ═════════════════════════════════════════════════════════════════════════════
# Degenerate inputs
# ═════════════════════════════════════════════════════════════════════════════
def test_no_chunks_returns_empty():
    assert retriever.score_chunks(CLAIM, []) == []


def test_claim_with_no_scoreable_tokens():
    """
    A claim of only short words has an empty token set; scoring must not
    divide by zero. Every chunk comes back at 0.0.
    """
    scored = retriever.score_chunks("It is on a us", ALL_CHUNKS)
    assert len(scored) == len(ALL_CHUNKS)
    assert all(s == 0.0 for s, _, _ in scored)


def test_single_chunk_does_not_break_idf():
    """
    With one chunk every term has df == 1, so the weights must stay positive
    rather than collapsing to log(1) == 0 and zeroing the denominator.
    """
    scored = retriever.score_chunks(CLAIM, [RELEVANT])
    assert len(scored) == 1
    assert scored[0][0] > 0.0


def test_chunk_sharing_no_words_scores_zero():
    unrelated = ("Photosynthesis converts light energy into chemical energy.", "u")
    scored = retriever.score_chunks(CLAIM, [RELEVANT, unrelated])
    assert {c: s for s, c, _ in scored}[unrelated[0]] == 0.0
