"""
Unit tests for Pipeline/ner.py.

Every case below is a bug that reached a running pipeline. The point of the
file is that none of them ever do so again silently.

Nothing in Pipeline/ is modified or mocked: these tests import the real module
and call the real functions. spaCy's en_core_web_sm loads once when pytest
imports ner, which costs a second or two for the whole session.

    cd Backend
    pytest                 # everything
    pytest -m "not slow"   # skip anything touching network or models
    pytest -k subject      # only tests with "subject" in the name
"""

import pytest

from Pipeline import ner


# ═════════════════════════════════════════════════════════════════════════════
# filter_entities — pure, no parser involved
# ═════════════════════════════════════════════════════════════════════════════
def test_date_is_dropped_when_a_real_anchor_exists():
    """
    'the beginning of the 17th century' once filled half of a two-term query
    and retrieved Wikipedia's *17th century BC* article, so a true claim came
    back NOT ENOUGH INFO. A date never helps a lookup when a named entity is
    available.
    """
    assert ner.filter_entities([
        ("The Republic of Sale", "GPE"),
        ("the beginning of the 17th century", "DATE"),
    ]) == ["The Republic of Sale"]


def test_date_survives_when_it_is_the_only_entity():
    """Dropping the last entity would leave the query builder with nothing."""
    assert ner.filter_entities([("1889", "DATE")]) == ["1889"]


def test_entities_are_ordered_by_priority():
    """LOC (4) outranks ORG (3) — which is exactly why subject promotion exists."""
    assert ner.filter_entities([
        ("Africa", "LOC"),
        ("Europe", "LOC"),
        ("Yavuz Sultan Selim Bridge", "ORG"),
    ]) == ["Africa", "Europe", "Yavuz Sultan Selim Bridge"]


def test_empty_entity_list():
    assert ner.filter_entities([]) == []


# ═════════════════════════════════════════════════════════════════════════════
# build_query / _sanitize_query — pure
# ═════════════════════════════════════════════════════════════════════════════
def test_query_is_capped_at_two_entities():
    assert ner.build_query(["A", "B", "C"], "") == "A B"


def test_no_entities_yields_no_query():
    """None is the signal that triggers keyword fallback; '' would not."""
    assert ner.build_query([], "") is None


def test_possessives_and_brackets_are_stripped():
    """"Gustave Eiffel's" must not become the garbage token "Gustave Eiffels"."""
    assert ner._sanitize_query("Gustave Eiffel's Tower (Paris)") == "Gustave Eiffel Tower Paris"


# ═════════════════════════════════════════════════════════════════════════════
# extract_subject — needs the dependency parser
# ═════════════════════════════════════════════════════════════════════════════
def test_subject_keeps_prepositional_attachment():
    """
    The regression that motivated _subject_span. spaCy's noun_chunks are *base*
    noun phrases and exclude PP attachments, so "The Republic of Sale" chunked
    as "The Republic" and the query was built from the bare word "Republic".

    Asserted as containment rather than an exact string: what matters is that
    "of Sale" survived, not the precise determiner handling.
    """
    subject = ner.extract_subject(
        "The Republic of Sale traces its origins back to the beginning of the 17th century."
    )
    assert subject is not None
    assert "Sale" in subject


def test_subject_found_for_bare_proper_noun():
    """
    NER misses an undetermined proper noun at sentence start; the parser does
    not, because identifying an nsubj needs syntax rather than world knowledge.
    """
    assert ner.extract_subject("Bosporus is located between Africa and Europe.") == "Bosporus"


def test_passive_subject_is_recognised():
    """Without nsubjpass, every 'X is located...' sentence returns None."""
    subject = ner.extract_subject("Yavuz Sultan Selim Bridge was located between Africa and Europe.")
    assert subject == "Yavuz Sultan Selim Bridge"


def test_leading_article_is_stripped():
    subject = ner.extract_subject("The Bosporus connects the Black Sea to the Sea of Marmara.")
    assert subject is not None
    assert not subject.lower().startswith("the ")


def test_pronoun_subject_is_rejected():
    """'You' is a valid subject and a useless search anchor."""
    assert ner.extract_subject("You can walk between them with the bridge.") is None


@pytest.mark.timeout(10)
def test_subject_extraction_terminates_on_relative_clause():
    """
    Guards a real hang. The first version of _subject_span walked up the tree
    with `while cur is not head`, but spaCy builds a fresh Token object on every
    attribute access, so identity never matched and the loop spun forever.
    A relative clause is the shape that exercised it.

    Requires pytest-timeout; without it this still passes, just without the guard.
    """
    ner.extract_subject("The bridge that connects Asia and Europe was built in 2016.")


# ═════════════════════════════════════════════════════════════════════════════
# check_decontextualized — AIDA "Independent"
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("claim, should_reject, why", [
    ("The Republic of Sale traces its origins to the 17th century.", False,
     "'its' is bound by the subject in the same clause — nothing to resolve"),
    ("They were expelled by the order of the Spanish king.", True,
     "'They' has no antecedent inside the claim"),
    ("It was first climbed in 1953 by Edmund Hillary.", True,
     "'It' leads the claim; needs the entity from the previous sentence"),
    ("You can walk between Africa and Europe.", False,
     "generic 'you' refers to nobody and can never be resolved"),
    ("The strait that connects the Black Sea is narrow.", False,
     "'that' is a relativiser, not a dangling reference"),
    ("Mount Everest was climbed by Hillary and he was later knighted.", False,
     "'he' follows candidate antecedents in the same claim"),
])
def test_decontextualization(claim, should_reject, why):
    rejected = ner.check_decontextualized(claim) is not None
    assert rejected is should_reject, why


# ═════════════════════════════════════════════════════════════════════════════
# check_fluency
# ═════════════════════════════════════════════════════════════════════════════
def test_well_formed_claim_passes_fluency():
    assert ner.check_fluency("The Bosporus connects the Black Sea to the Sea of Marmara.") is None


@pytest.mark.parametrize("claim", [
    "Bosporus strait Turkey",                 # no main verb
    "Is the Bosporus in Europe?",             # a question, not a statement
    "The Bosporus connects the",              # truncated mid-phrase
    "The Bosporus connects the.",             # same, with a trailing full stop
    "The Bosporus is located between",        # dangling preposition
    "The Bosporus connects the Black Sea and",  # dangling conjunction
])
def test_malformed_claims_are_caught(claim):
    assert ner.check_fluency(claim) is not None


@pytest.mark.parametrize("claim", [
    "The Bosporus connects the Black Sea to the Sea of Marmara.",
    "Bosporus is located between Africa and Europe.",
    "The Republic of Sale traces its origins to the 17th century.",
    "The Bosporus is 31 kilometres long.",
])
def test_valid_claims_are_not_flagged_as_truncated(claim):
    """
    The closed-word list must not fire on well-formed claims. Real claims end
    in nouns, adjectives or numbers — never in a word that requires a
    complement — so this should be a comfortable margin, but it is the failure
    mode a lexical check would have if the list were drawn too widely.
    """
    assert ner.check_fluency(claim) is None


def test_fluency_failure_explains_itself():
    """The message is fed back into the retry prompt, so it must be readable."""
    defect = ner.check_fluency("Bosporus strait Turkey")
    assert isinstance(defect, str) and len(defect) > 10


# ═════════════════════════════════════════════════════════════════════════════
# check_worthy
# ═════════════════════════════════════════════════════════════════════════════
def test_opinion_without_any_entity_is_filtered():
    """Costs ~152s to discover the expensive way, and always returns NEI."""
    assert ner.check_worthy("The weather seems wonderful and the food tastes amazing.") is not None


@pytest.mark.parametrize("claim", [
    "Istanbul is the most populous city in Turkey.",   # evaluative wording, still checkable
    "The Bosporus is 31 kilometres long.",
])
def test_claims_with_a_named_entity_are_always_admitted(claim):
    """Conservative by design: a false reject silently loses a checkable claim."""
    assert ner.check_worthy(claim) is None


# ═════════════════════════════════════════════════════════════════════════════
# split_sentences
# ═════════════════════════════════════════════════════════════════════════════
def test_two_sentences_are_split():
    assert len(ner.split_sentences(
        "Bosporus is located between Africa and Europe. "
        "You can walk between them with the bridge."
    )) == 2


def test_decimal_point_does_not_split_a_sentence():
    """Why this uses the parser and not a regex on '.'."""
    assert len(ner.split_sentences("The strait is 3.7 km long at its narrowest point.")) == 1


# ═════════════════════════════════════════════════════════════════════════════
# extract_queries — the whole NER path end to end (no network)
# ═════════════════════════════════════════════════════════════════════════════
def test_subject_is_not_evicted_by_its_own_fragment():
    """
    Promotion once made retrieval *worse*: the truncated subject 'Republic'
    matched the entity 'The Republic of Sale' by substring, so the good entity
    was dropped as a duplicate and the query was built from the fragment.
    """
    queries = ner.extract_queries(
        "The Republic of Sale traces its origins back to the beginning of the 17th century."
    )
    assert any("Sale" in q for q in queries), queries
    assert not any("17th century" in q for q in queries), queries


def test_buried_subject_reaches_the_query():
    """
    'Yavuz Sultan Selim Bridge' is tagged ORG (priority 3) and loses to Africa
    and Europe (LOC, 4), so the top-2 cap discarded the only entity that could
    retrieve a relevant article.
    """
    queries = ner.extract_queries(
        "Yavuz Sultan Selim Bridge was located between Africa and Europe."
    )
    assert any("Yavuz" in q for q in queries), queries


def test_query_set_is_deduplicated():
    queries = ner.extract_queries("Bosporus is located between Africa and Europe.")
    assert len(queries) == len(set(queries))


# ═════════════════════════════════════════════════════════════════════════════
# check_negation — polarity
# ═════════════════════════════════════════════════════════════════════════════
def test_inserted_negation_is_rejected():
    """
    THE FABRICATION CASE, verbatim from a run.

    The extractor turned a positive claim about the Amazon into a NEGATIVE claim
    about the Nile. Every existing gate passed it — each content word appears in
    the source, and 'not' is three characters so MIN_CONTENT_WORD_LENGTH filters
    it out before the faithfulness comparison ever runs. The verifier then
    refuted the fabrication and the pipeline reported a confident REFUTES on a
    claim the input never made.
    """
    source = ("It is universally acknowledged as longer than the Nile by all "
              "international cartographers.")
    claim = "The Nile is not universally acknowledged as the world's longest river."
    assert ner.check_negation(source, claim) is not None


def test_dropped_negation_is_rejected():
    """The other direction. Removing a 'not' reverses meaning just as thoroughly."""
    source = "The colony was not abandoned in 1610."
    claim = "The colony was abandoned in 1610."
    assert ner.check_negation(source, claim) is not None


def test_matching_polarity_passes():
    """A faithful rewrite that preserves polarity must not be rejected."""
    source = "The museum opened in 1932 and holds over 400 paintings."
    claim = "The museum opened in 1932."
    assert ner.check_negation(source, claim) is None


def test_preserved_negation_passes():
    """A negation carried through faithfully is fine — the gate is not anti-'not'."""
    source = "The bridge was not completed until 1973."
    claim = "The bridge was not completed until 1973."
    assert ner.check_negation(source, claim) is None


def test_negative_quantifier_counts_as_negation():
    """
    'no other' carries negation with no `neg` dependency arc anywhere, which is
    why the lexicon exists alongside the dependency test. Neither check alone
    covers the class.
    """
    assert ner._negation_count("No other river is longer.") >= 1
    assert ner._negation_count("Every other river is shorter.") == 0
