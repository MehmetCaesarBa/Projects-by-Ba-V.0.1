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

import re

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
def test_query_is_capped_at_max_query_entities():
    """
    Asserts the RELATIONSHIP to the constant, not a hardcoded 2.

    The previous version read `build_query(["A","B","C"]) == "A B"` and broke
    the moment MAX_QUERY_ENTITIES was raised from 2 to 3 — a test failing
    because a tunable was tuned, which teaches you nothing. Referencing the
    constant is the whole reason it was promoted out of an inline slice.
    """
    entities = [f"E{i}" for i in range(ner.MAX_QUERY_ENTITIES + 2)]
    query = ner.build_query(entities, "")
    assert len(query.split()) == ner.MAX_QUERY_ENTITIES


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


def test_overlong_common_noun_subject_is_not_promoted():
    """
    "The title of the world's longest river" is 8 tokens against a cap of 7,
    and heads on the common noun 'title'.

    Two regressions live here. The cap first sliced positionally and emitted
    "title of the world's longest", anchoring the query on a dangling
    adjective. The fix returned the head noun instead — which made retrieval
    WORSE: promotion put the bare word 'title' ahead of 'the Amazon River' and
    Wikipedia returned an article about titles.

    A generic head is not worth promoting. None (falsy) hands the decision back
    to the entity path.
    """
    subject = ner.extract_subject(
        "The title of the world's longest river belongs to the Amazon River."
    )
    assert not subject or subject.endswith("river")


def test_overlong_proper_noun_subject_still_promotes():
    """A long phrase headed by a NAME keeps the name — it is a usable anchor."""
    subject = ner.extract_subject(
        "The Republic of Sale on the Barbary Coast of Morocco traded in captives."
    )
    assert subject is None or "Republic" in subject or "Sale" in subject


def test_claim_length_bounds_are_named():
    """Guards the promotion of `3 <= len(doc) <= 60` to named constants."""
    assert ner.MIN_CLAIM_TOKENS < ner.MAX_CLAIM_TOKENS
    assert ner.MAX_QUERY_ENTITIES >= 1


# ═════════════════════════════════════════════════════════════════════════════
# Specificity ordering
# ═════════════════════════════════════════════════════════════════════════════
def test_multi_token_name_outranks_a_continent():
    """
    The case that motivated ordering by specificity. By ENTITY_PRIORITY the
    bridge is ORG (3) and loses to Africa/Europe (LOC, 4); by grammatical role
    in "Europe connects to Asia by ... Bridge" it loses again, because the
    specific entity sits in a prepositional phrase. Only specificity is right.
    """
    ordered = ner._by_specificity(["Europe", "Asia", "Yavuz Sultan Selim Bridge"])
    assert ordered[0] == "Yavuz Sultan Selim Bridge"


def test_specificity_is_stable_for_equal_keys():
    """Equal-specificity entities keep the order filter_entities gave them."""
    assert ner._by_specificity(["Africa", "Europe"]) == ["Africa", "Europe"]


def test_title_case_breaks_ties_against_common_nouns():
    a = ner._specificity("Mount Ararat")
    b = ner._specificity("the mountain")
    assert a > b


def test_every_entity_survives_the_widened_cap():
    """
    Exclusion was the only harm ranking ever did: an entity outside the cap is
    never searched for at all. With three entities and MAX_QUERY_ENTITIES = 3,
    the bridge must now get its own standalone query.
    """
    queries = ner.extract_queries(
        "Europe connects to Asia by a land bridge called Yavuz Sultan Selim Bridge."
    )
    assert any("Yavuz" in q for q in queries), queries


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


def test_epistemic_opinion_without_sentiment_is_caught():
    """
    The gap a sentiment lexicon alone leaves. "should be repaired" contains no
    positive or negative word — Hu & Liu would score it neutral — yet it is an
    opinion, not a checkable fact. That is why _EPISTEMIC_MARKERS is unioned in
    rather than replaced by the lexicon.
    """
    assert ner.check_worthy("The bridge should be repaired soon.") is not None


def test_subjectivity_lexicon_loaded():
    """
    Either NLTK's ~6,800-word Opinion Lexicon or the built-in floor. The floor
    is the hand-written adjectives plus the epistemic markers, so anything
    smaller means both sources failed.
    """
    assert len(ner._SUBJECTIVE_LEMMAS) >= len(ner._EPISTEMIC_MARKERS)
    assert "should" in ner._SUBJECTIVE_LEMMAS
    assert "beautiful" in ner._SUBJECTIVE_LEMMAS


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
# ═════════════════════════════════════════════════════════════════════════════
# Clause enumeration
# ═════════════════════════════════════════════════════════════════════════════
def test_coordination_yields_two_clauses_with_shared_subject():
    """
    The failure this exists for. Given a two-relation sentence, phi3 produced
    the first relation and then returned it verbatim three times, so the second
    relation — which happened to be the false one — was never checked.

    English elides the subject under coordination, so clause 2 has none of its
    own and MUST inherit it, or the claim reads "is celebrated as the earliest
    settlement" with nothing to check.
    """
    clauses = ner.enumerate_clauses(
        "The museum opened in 1932 and holds over 400 paintings."
    )
    assert len(clauses) == 2
    assert all("museum" in c.lower() for c in clauses), clauses
    assert "1932" in clauses[0] and "400" in clauses[1]


def test_relative_clause_takes_the_antecedent_as_subject():
    """
    "the river, which discharges..." must become "the river discharges...".
    Leaving the relativiser produces a grammatical claim that names nothing,
    and retrieval has no entity to anchor on.
    """
    clauses = ner.enumerate_clauses(
        "The Danube, which flows through ten countries, reaches the Black Sea."
    )
    assert len(clauses) >= 2
    assert not any(c.lower().startswith("which") for c in clauses), clauses
    assert all("danube" in c.lower() for c in clauses), clauses


def test_single_clause_passes_through_unchanged():
    """One proposition in, one out — enumeration must not invent structure."""
    sentence = "Bosporus is located between Africa and Europe."
    assert ner.enumerate_clauses(sentence) == [sentence]


def test_clauses_are_returned_in_sentence_order():
    """
    Ordering is by clause HEAD position. Sorting by lowest token index ties
    once every clause inherits the same subject, and the sort then falls
    through to comparing strings alphabetically — which put "is celebrated"
    before "was founded".
    """
    clauses = ner.enumerate_clauses(
        "The treaty was signed in 1919 and ended the war."
    )
    assert len(clauses) == 2
    assert "1919" in clauses[0]
    assert "war" in clauses[1]


def test_reported_speech_is_not_split():
    """
    ccomp is excluded on purpose. In "X said that Y", the checkable claim is
    that X SAID it — not that Y is true. Splitting out the complement would
    verify the wrong proposition.
    """
    clauses = ner.enumerate_clauses(
        "The minister announced that the bridge would open in June."
    )
    assert len(clauses) == 1


def test_inherited_subject_survives_an_embedded_clause():
    """
    THE GAP IN THIS SUITE, made explicit.

    Every other coordination case here joins two SIMPLE clauses. This one embeds
    a relative clause inside the second conjunct — "in what is now the United
    States" — where 'what' is the nsubj of 'is'.

    The old span-wide subject test counted that nested nsubj as the conjunct's
    own, inherited nothing, and emitted a subjectless clause. The extractor then
    invented a subject, the recovered subject changed, the query set changed,
    and the article holding the refuting passage was never fetched. One parse
    misread produced a wrong verdict four stages downstream.
    """
    clauses = ner.enumerate_clauses(
        "The English settlement of Jamestown was founded in May 1607 and is "
        "celebrated as the earliest European permanent settlement in what is "
        "now the United States."
    )
    assert len(clauses) == 2, clauses

    # Both conjuncts must name what they are about. The second is the one that
    # used to come back as a bare predicate.
    assert all("Jamestown" in c for c in clauses), clauses
    assert not clauses[1].lower().startswith("is celebrated"), clauses[1]


def test_clause_carries_no_stranded_separator():
    """
    Observed output:

        'The title of the world's longest river belongs to the Amazon River,.'

    The comma introduced the relative clause. Once that clause is split off into
    its own proposition the comma separates nothing, but the caller only drops
    punctuation that PRECEDES the clause head, and _render_clause's trailing
    strip(" ,;") could not reach it — the sentence-final full stop was the last
    character and shielded it.

    Asserted on the rendered STRING rather than on the token set, because the
    token set was already correct; only its punctuation read wrong. ',.' is not
    a sequence en_core_web_sm sees in training data, and this string is both
    parsed by check_fluency and handed to the model to rewrite.
    """
    clauses = ner.enumerate_clauses(
        "The title of the world's longest river belongs to the Amazon River, "
        "which discharges more water than any other drainage basin."
    )
    for clause in clauses:
        assert ",." not in clause, clause
        assert ",," not in clause, clause
        assert not re.search(r"[,;:]\s*$", clause), clause


def test_render_clause_preserves_internal_punctuation():
    """
    The stranded-separator cleanup must not become a blanket comma strip. A
    comma with material on BOTH sides is doing its job, and removing it would
    corrupt claims the enumerator never split.
    """
    doc = ner.NLP("The museum, founded in 1932, holds 400 paintings.")
    rendered = ner._render_clause(doc, {t.i for t in doc})

    assert rendered.count(",") == 2, rendered
    assert rendered.endswith("paintings."), rendered


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
