import re
import spacy
import pytest

# ── spaCy model ───────────────────────────────────────────────────────────────
# Run once to download: python -m spacy download en_core_web_sm
NLP = spacy.load("en_core_web_sm")

# ── Entity priority for query construction ────────────────────────────────────
# The weights are assigned based on semantic density and historical retrieval value.
# Higher index = higher priority for anchoring a search index query.
ENTITY_PRIORITY = {
    # 1. DATE: Lowest anchor value. Chronological markers provide no contextual meaning
    #    unless paired with a concrete event, subject, or organization.
    "DATE"   : 1,
    
    # 2. NORP: Nationalities, religious groups, or political factions. 
    #    Useful for macro-demographics but too broad on their own for single-hop retrieval.
    "NORP"   : 2,
    
    # 3. ORG: Companies, institutions, or open-source engineering groups (e.g., "OpenAI", "Apache").
    #    Acts as an excellent contextual domain wrapper for factual claims.
    "ORG"    : 3,

    # 3b. PRODUCT / WORK_OF_ART / LOC: Named objects, creative works, and
    #     non-GPE locations (mountain ranges, regions). Mid-priority anchors.
    "PRODUCT"     : 3,
    "WORK_OF_ART" : 4,
    "LOC"         : 4,

    # 4. GPE: Geopolitical entities (countries, cities, sovereign states).
    #    Provides an immediate geographical constraint to eliminate out-of-domain noise.
    "GPE"    : 4,

    # 5. EVENT: Named battles, tech conferences, or historic occurrences (e.g., "WWII", "ROSCon").
    #    Strongly binds specific timeline occurrences together.
    "EVENT"  : 5,

    # 5b. FAC: Named facilities — buildings, monuments, bridges, airports
    #     (e.g., "Eiffel Tower", "Statue of Liberty"). Previously missing,
    #     which made NER blind to the very landmarks test claims are about:
    #     spaCy tags them FAC, not GPE/ORG, so only the DATE survived and
    #     every landmark query fell through to keyword fallback.
    "FAC"    : 5,

    # 6. PERSON: Individual historical figures, authors, or developers.
    #    Highest priority because unique proper names are the strongest search anchors in knowledge bases.
    "PERSON" : 6
}

# ── spaCy stopwords for fallback keyword extraction ───────────────────────────
STOPWORDS = NLP.Defaults.stop_words


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Named Entity Recognition
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities(fact: str) -> list[tuple[str, str]]:
    """
    Runs spaCy NER on an isolated atomic fact.
    Filters the results to retain only the specific categories mapped in 
    the AFEV search-anchoring hierarchy (ENTITY_PRIORITY).

    Entity Classifications Explained:
    - PERSON: Unique human agents (e.g., "Alan Turing")
    - EVENT: Fixed, named historical occurrences (e.g., "Industrial Revolution")
    - GPE: Geopolitical map coordinates (e.g., "California")
    - ORG: Bound organizational bodies (e.g., "MIT")
    - NORP: Broad human group associations (e.g., "Byzantines")
    - DATE: Specific temporal constraints (e.g., "2026")
    """
    doc = NLP(fact)

    entities = [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in ENTITY_PRIORITY
    ]

    print(f"[NER] Extracted entities: {entities}")
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Entity Filtering and Prioritization
# ─────────────────────────────────────────────────────────────────────────────
def filter_entities(entities: list[tuple[str, str]]) -> list[str]:
    """
    Sorts extracted entities by structural priority and enforces grounding rules.
    
    Grounding Rule (Date Isolation):
    A standalone 'DATE' entity (e.g., "in 1995") possesses zero unique semantic 
    meaning in a dense vector database or standard token search without context. 
    Therefore, a DATE entity is dropped entirely unless accompanied by an entity 
    with a priority score >= 2 (NORP, ORG, GPE, EVENT, or PERSON).

    Example:
        Fact: "Python was conceived by Guido van Rossum in 1989"
        Parsed: [("Guido van Rossum", "PERSON"), ("Python", "ORG"), ("1989", "DATE")]
        Sorted Output: ["Guido van Rossum", "Python", "1989"]
    """
    if not entities:
        return []

    # Sort entities based on their explicit knowledge-base retrieval weights (Descending)
    sorted_entities = sorted(
        entities,
        key=lambda e: ENTITY_PRIORITY.get(e[1], 0),
        reverse=True
    )

    # A DATE is dropped whenever ANY other entity survives to anchor the query.
    #
    # The rule used to be the opposite way round — a date was kept as soon as a
    # high-priority entity existed — which let "the beginning of the 17th
    # century" occupy half of a two-term query and retrieve the *17th century
    # BC* article. A date contributes nothing to a Wikipedia lookup even beside
    # a good anchor: it either matches the year article (noise) or narrows
    # nothing. It is retained only when it is the sole entity available, so the
    # keyword fallback still has something to work with.
    has_non_date = any(label != "DATE" for _, label in sorted_entities)

    filtered = []
    for text, label in sorted_entities:
        if label == "DATE" and has_non_date:
            print(f"[NER] Dropping date entity from query: '{text}'")
            continue
        filtered.append(text)

    print(f"[NER] Filtered entities: {filtered}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0b — Sentence Segmentation
# ─────────────────────────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """
    Split input text into sentences using spaCy's parser-driven segmentation.

    Used by the extractor to give each sentence its own decomposition budget.
    Previously the loop ran once over the whole document, so a sentence the
    model refused to decompose faithfully consumed every remaining iteration
    and starved everything after it — an observed run spent all its retries
    arguing about sentence 1 and never looked at sentence 2 at all.

    Parser-based segmentation (not a regex on '.') because abbreviations,
    decimals and initials make period-splitting wrong often enough to matter:
    "Bosporus is 3.7 km long. It is narrow." must not become three sentences.
    """
    doc = NLP(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    print(f"[NER] Segmented input into {len(sentences)} sentence(s).")
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0b2 — Clause Enumeration
# ─────────────────────────────────────────────────────────────────────────────
# Relations that introduce a SEPARATE proposition worth checking on its own.
#
#   conj    coordination — "X was founded in 1607 AND is celebrated as ..."
#   relcl   relative clause — "the Amazon, WHICH discharges more water than ..."
#   advcl   adverbial clause — "BECAUSE it rose so high, it surpassed ..."
#
# ccomp is deliberately EXCLUDED. In "Stalin confessed that he intended to
# install elections", the ccomp is "he intended to install elections" — but the
# claim being made is that Stalin SAID it, not that it is true. Extracting the
# complement as a standalone fact would check the wrong proposition.
_CLAUSE_DEPS = ("conj", "relcl", "advcl")

# Below this many tokens a clause is a fragment, not a proposition.
MIN_CLAUSE_TOKENS = 3

# Words that stand in for a noun inside a relative clause. Matched by surface
# form as well as by tag, because a mis-tagged relativiser left in place yields
# a claim like "which discharges more water" — grammatical, and unanchorable
# for retrieval since it names nothing.
_RELATIVISERS = {"which", "who", "whom", "whose", "that"}


def _clause_head_tokens(doc) -> list:
    """
    Every token that heads its own proposition, in sentence order.

    Starts at the ROOT and follows _CLAUSE_DEPS transitively, so a chain like
    "A happened, B happened and C happened" yields three heads rather than two.
    Only verbs and auxiliaries qualify — a coordinated NOUN ("Peter and Paul
    travelled") is one proposition with a compound subject, not two.
    """
    root = next((t for t in doc if t.head == t), None)
    if root is None:
        return []

    # Scanned across the whole doc rather than descended from the ROOT, because
    # a relative clause attaches to a NOUN, not to a verb: in "the Amazon,
    # which discharges more water", `discharges` is a relcl child of `Amazon`.
    # Walking only verb-to-verb never reaches it, and the proposition is lost.
    heads = [root] + [
        t for t in doc
        if t.i != root.i and t.dep_ in _CLAUSE_DEPS and t.pos_ in ("VERB", "AUX")
    ]
    return sorted(heads, key=lambda t: t.i)


def _inherited_subject(head):
    """
    The subject a clause borrows when it has none of its own.

    English elides the subject under coordination: in "X was founded in 1607
    and is celebrated as ...", the second clause has no nsubj at all. Without
    restoring it the clause reads "is celebrated as the earliest ...", which is
    not a checkable claim — it has no subject to check.

    For a relative clause the antecedent is the noun the clause modifies:
    "the Amazon, which discharges ..." -> subject is "the Amazon".

    "HAS ITS OWN SUBJECT" MEANS THE HEAD'S OWN CHILD, NOT ANY TOKEN IN THE SPAN.
    This test used to scan every token in the clause's index set:

        any(t.dep_ in ("nsubj", ...) for t in head.doc if t.i in own_indices)

    which counts subjects belonging to NESTED clauses. Observed on the project's
    own test input:

        "...was founded in May 1607 and is celebrated as the earliest European
         permanent settlement in what is now the United States."

    The second clause contains "in WHAT IS now the United States", and 'what' is
    the nsubj of 'is'. The scan found it, concluded 'celebrated' already had a
    subject, and inherited nothing. The clause came out as

        'is celebrated as the earliest European permanent settlement in what is
         now the United States'

    with no subject at all — and the model then invented one, producing a claim
    about a bare "Jamestown" instead of "The English settlement of Jamestown".
    That changed the recovered subject, which changed the query set, which
    changed the article pool, which is where the passage that refutes the claim
    stopped being retrieved. A parse-level false negative propagated all the way
    to the verdict.

    Subjecthood is a relation between a token and ITS head, so the question is
    only ever about head.children. Scanning the span asks a different question
    and gets a different answer whenever the clause embeds another clause.

    WHY NO TEST CAUGHT IT: every clause case in the suite coordinates two SIMPLE
    clauses ("The museum opened in 1932 and holds over 400 paintings"). None
    embeds a subordinate clause inside the second conjunct, which is the only
    shape that triggers the bug.
    """
    if any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in head.children):
        return None                                  # it has its own

    if head.dep_ == "relcl":
        return head.head                             # the modified noun

    # Walk up the clause chain looking for a subject to share.
    cur = head
    while cur.head.i != cur.i:
        cur = cur.head
        for child in cur.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
    return None


def _render_clause(doc, indices: set[int]) -> str:
    """
    Turn a set of token indices back into readable text.

    Joining tokens with spaces reintroduces the spacing the tokenizer removed:
    "founded in May 1607 ." and "Jamestown 's founding". The clause is fed
    straight to an LLM and to the faithfulness gate, so it has to read as a
    sentence rather than as a token dump.

    STRANDED SEPARATORS. The caller drops punctuation only when it precedes the
    clause head (`t.is_punct and t.i < head.i`), so a comma that came AFTER the
    head survives into the clause even though the material it separated does
    not. Observed:

        'The title of the world's longest river belongs to the Amazon River,.'

    The comma introduced the relative clause, which now belongs to a different
    clause entirely; what remains is a separator with nothing left to separate.
    The trailing `.strip(" ,;")` below cannot reach it, because the sentence-final
    full stop is the last character and shields it from the strip.

    Cleaned here rather than in the caller because it is a rendering concern:
    the token SET is right, only its punctuation reads wrong. It matters beyond
    tidiness — the clause string is what check_fluency parses and what the model
    is asked to rewrite, and ',.' is not a sequence en_core_web_sm sees in
    training data.
    """
    text = " ".join(doc[i].text for i in sorted(indices))
    text = re.sub(r"\s+([.,;:!?)\]])", r"\1", text)   # no space before closers
    text = re.sub(r"([(\[])\s+", r"\1", text)         # no space after openers
    text = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d)\b", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)

    # A separator immediately followed by another separator, or sitting at the
    # very end, has nothing left on one side of it. Runs first so that ',.'
    # becomes '.' before the trailing strip runs.
    text = re.sub(r"[,;:]+(?=\s*[,;:.!?])", "", text)
    text = re.sub(r"[,;:]+\s*$", "", text)

    return text.strip(" ,;")


def enumerate_clauses(sentence: str) -> list[str]:
    """
    Split a sentence into its clause-level propositions, one per predicate.

    WHY THIS EXISTS: the extractor used to be asked "read this sentence, diff it
    against the facts you already produced, and give me the next uncovered
    relation." That is set-difference over semantics, and phi3:mini cannot do
    it — given a two-clause sentence it produced the first relation and then
    returned it verbatim three times in a row, so the second relation (which
    happened to be the FALSE one) was never extracted or checked.

    Syntax answers the same question mechanically. The clauses are marked in the
    parse; no search is required. The model's job shrinks from "find what is
    missing" to "tidy this clause into a sentence", which it is good at.

    Returns the clause strings in sentence order, each with a subject restored
    if it borrowed one. Returns [sentence] unchanged when enumeration finds
    nothing better — a single-clause sentence is already one proposition, and a
    parse failure should degrade to current behaviour rather than lose the
    sentence.
    """
    doc = NLP(sentence)
    heads = _clause_head_tokens(doc)

    if len(heads) < 2:
        return [sentence.strip()]

    head_indices = {h.i for h in heads}
    clauses: list[tuple[int, str]] = []

    for head in heads:
        # A clause owns its subtree MINUS the subtrees of the other clause
        # heads, so coordinated material is not duplicated across clauses.
        own = {t.i for t in head.subtree}
        for other in heads:
            if other.i != head.i and other.i in own:
                own -= {t.i for t in other.subtree}

        if head.dep_ == "relcl":
            # "the Amazon, WHICH discharges more water" -> the proposition is
            # "the Amazon discharges more water". Swap the relativiser for the
            # noun it stands in for; leaving 'which' produces a claim with no
            # identifiable subject, which retrieval cannot anchor on.
            own -= {
                t.i for t in doc
                if t.i in own and (
                    t.tag_ in ("WDT", "WP", "WP$")
                    or (t.pos_ == "PRON" and t.text.lower() in _RELATIVISERS)
                )
            }
            own |= ({t.i for t in head.head.subtree}
                    - {t.i for t in head.subtree})
        else:
            subject = _inherited_subject(head)
            if subject is not None:
                own |= {t.i for t in subject.subtree}

        # Punctuation and conjunctions are stripped LAST, after the subject or
        # antecedent has been merged in. Doing it earlier let the antecedent's
        # subtree reintroduce the comma that separated the relative clause:
        # "The Amazon River, discharges more water".
        own -= {
            t.i for t in doc
            if t.i in own and (t.dep_ == "cc" or (t.is_punct and t.i < head.i))
        }

        text = _render_clause(doc, own)
        if len(own) >= MIN_CLAUSE_TOKENS and text:
            # Ordered by the HEAD's position, not by the lowest token index.
            # Once a subject is inherited, every clause starts at index 0, so
            # min(own) ties and the sort falls through to comparing the strings
            # alphabetically — which put "is celebrated" before "was founded".
            clauses.append((head.i, text))

    if not clauses:
        return [sentence.strip()]

    ordered = [text for _, text in sorted(clauses)]
    print(f"[NER] Sentence split into {len(ordered)} clause(s).")
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0c — Claim Quality Gates (Fluency + Check-worthiness)
# ─────────────────────────────────────────────────────────────────────────────
# Both gates run on a parse we already need, so they cost ~10ms and no model
# call. That matters: admitting one claim to retrieval + verification costs
# ~152s measured, so a gate priced in milliseconds pays for itself the first
# time it fires, whereas an LLM-based check-worthiness classifier at ~10s per
# claim only breaks even if it rejects more than ~6.6% of claims.

# ── Subjectivity lexicon ──────────────────────────────────────────────────────
# SENTIMENT IS NOT SUBJECTIVITY, and the gate needs both.
#
# Hu & Liu's Opinion Lexicon (~6,800 words, bundled with NLTK) covers evaluative
# vocabulary: beautiful, terrible, stunning, awful. It does NOT cover epistemic
# and deontic markers — 'seems', 'believes', 'should', 'ought' — which make a
# sentence subjective while carrying no sentiment at all. "The bridge should be
# repaired" contains no positive or negative word and is still an opinion.
#
# So the lexicon is the union of the two: an off-the-shelf sentiment list for
# breadth, plus a hand-written set for the modal/epistemic class that sentiment
# resources systematically miss.
#
# Falls back to the original hand-written set when NLTK or its corpus is
# unavailable, so a fresh clone works before anyone runs nltk.download().

# Epistemic and deontic markers. Closed class, and absent from every sentiment
# lexicon — these are the words that make a claim an assertion about the
# speaker's state rather than about the world.
_EPISTEMIC_MARKERS = {
    "seem", "feel", "believe", "think", "suppose", "guess", "reckon",
    "deserve", "should", "ought", "must", "probably", "arguably",
    "apparently", "presumably", "allegedly", "supposedly",
}

# Retained verbatim as the fallback, and as a floor: even with NLTK present
# these must be treated as subjective.
_HAND_WRITTEN_SUBJECTIVE = {
    "beautiful", "ugly", "best", "worst", "greatest", "amazing", "terrible",
    "wonderful", "awful", "boring", "interesting", "important", "nice",
    "better", "worse", "favourite", "favorite", "stunning", "lovely",
}


def _load_subjective_lexicon() -> set[str]:
    """
    Opinion Lexicon ∪ epistemic markers, or the hand-written floor if NLTK is
    not installed. Loaded once at import; the corpus is ~200 KB of plain text.

        python -c "import nltk; nltk.download('opinion_lexicon')"
    """
    lexicon = set(_HAND_WRITTEN_SUBJECTIVE) | _EPISTEMIC_MARKERS

    try:
        from nltk.corpus import opinion_lexicon
        words = set(opinion_lexicon.words())      # raises if corpus is missing
        lexicon |= {w.lower() for w in words}
        print(f"[NER] Subjectivity lexicon: {len(lexicon)} words "
              f"(NLTK opinion_lexicon + epistemic markers).")
    except Exception as e:
        print(f"[NER] NLTK opinion_lexicon unavailable ({type(e).__name__}); "
              f"using the built-in {len(lexicon)}-word list. "
              f"Run: python -c \"import nltk; nltk.download('opinion_lexicon')\"")

    return lexicon


_SUBJECTIVE_LEMMAS = _load_subjective_lexicon()


# A sentence ending in one of these is cut off, whatever the tagger decided.
# Determiners, prepositions, conjunctions and relativisers all REQUIRE a
# following complement; none can legitimately end a declarative claim.
_TRUNCATING_FINAL_WORDS = {
    # determiners
    "the", "a", "an", "this", "that", "these", "those", "its", "their", "his",
    "her", "our", "your", "my",
    # prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "onto",
    "about", "between", "among", "through", "during", "over", "under",
    # conjunctions and relativisers
    "and", "or", "but", "which", "who", "whom", "whose", "because", "while",
}


# Length bounds for a single atomic claim, previously the inline `3 <= len(doc)
# <= 60`. The lower bound rejects fragments; the upper bound is a proxy for
# atomicity — a 60-token "claim" is a paragraph and will contain several
# relations, which defeats the point of decomposition. Both are cheap to sweep
# offline: run check_fluency over a labelled set and count false rejections.
MIN_CLAIM_TOKENS = 3
MAX_CLAIM_TOKENS = 60


def check_fluency(claim: str) -> str | None:
    """
    Is this a well-formed declarative sentence? (Ullrich et al., 2025, §4.1)

    Returns None when fluent, otherwise a plain-language defect string that the
    extractor feeds back into its retry prompt.

    Deliberately BINARY and structural rather than a perplexity/SLOR score with
    a tuned cutoff. A continuous score needs a threshold calibrated against
    labelled data that does not exist yet, and would drift; these checks are
    facts about the parse tree, identical on every run and free to compute.
    Keep SLOR for offline evaluation, where a continuous scale is the point.
    """
    doc = NLP(claim)

    root = next((t for t in doc if t.head == t), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return "it has no main verb — a claim must be a full sentence, not a fragment"

    if not any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in doc):
        return "it has no subject — state who or what the claim is about"

    if claim.rstrip().endswith("?"):
        return "it is a question, not a statement"

    # Truncation is checked LEXICALLY as well as by part of speech, because the
    # POS branch alone cannot be trusted here. A tagger is trained on
    # well-formed sentences, so its output is least reliable on exactly the
    # malformed input this gate exists to catch: given "The Bosporus connects
    # the", en_core_web_sm does not label the stranded "the" as DET — it fits
    # the slot after a transitive verb and gets tagged as a nominal instead.
    # The check silently passed and a truncated claim was admitted.
    #
    # Function words are a CLOSED CLASS — a finite list that gains no new
    # members — so an explicit set is not a heuristic standing in for a model,
    # it is the complete answer. No tagger can beat a lookup table here.
    # Skip trailing punctuation: a model that emits "…connects the." is just as
    # truncated as one that emits "…connects the", and doc[-1] would otherwise
    # be the full stop.
    last = next((t for t in reversed(doc) if not t.is_punct), None)
    if last is not None and (
        last.pos_ in ("ADP", "CCONJ", "SCONJ", "DET")
        or last.text.lower() in _TRUNCATING_FINAL_WORDS
    ):
        return "it ends mid-phrase and appears truncated"

    if not (MIN_CLAIM_TOKENS <= len(doc) <= MAX_CLAIM_TOKENS):
        return "it is too short or too long to be a single atomic claim"

    return None


# Pronouns that point BACK at something said earlier. A claim containing one
# of these is not intelligible on its own, and its antecedent is recoverable
# from the source document — so the extractor can and should resolve it.
_ANAPHORIC = {
    "it", "its", "they", "them", "their", "theirs",
    "he", "him", "his", "she", "her", "hers",
    "this", "that", "these", "those", "there", "then",
}

# Pronouns with NO antecedent anywhere in the document — generic or deictic
# usage ("you can walk across it" means "one can"). These cannot be resolved
# by any amount of rereading, so demanding resolution would send the extractor
# into a rejection loop it can never escape. They are left alone; the query
# path already declines to anchor on a pronoun subject.
_GENERIC_PRONOUNS = {"you", "your", "one", "we", "our", "us", "i", "my"}


def check_decontextualized(claim: str) -> str | None:
    """
    Is the claim intelligible standing alone? (AIDA's 'Independent', via
    Wright et al. 2022 / Ullrich et al. 2025.)

    Returns None when self-contained, otherwise the unresolved reference.

    This is the metric the pipeline was missing, and its absence was visible:
    "You can walk between them with Yavuz Sultan Selim Bridge" reached
    retrieval with 'them' unresolved on the first pass. The fix is not to
    forbid pronouns — generic 'you' has no referent to recover — but to
    require that ANAPHORIC ones be replaced with wording from the document.

    Relativisers are exempt: "the strait that connects the Black Sea" is a
    perfectly self-contained claim, and its 'that' is grammatical machinery
    rather than a dangling reference. spaCy tags those WDT/WP, which is how
    they are told apart from the demonstrative 'that'.
    """
    doc = NLP(claim)

    # A pronoun is only a decontextualization failure when its antecedent lies
    # OUTSIDE the claim. Two cases that look alike and are not:
    #
    #   "The Republic of Sale traces ITS origins to the 17th century."
    #        -> 'its' is bound by the subject in the same clause. The claim is
    #           perfectly intelligible alone. Nothing to resolve.
    #
    #   "THEY were expelled by the order of the Spanish king."
    #        -> nothing in the claim says who. Genuinely broken standalone.
    #
    # The earlier version rejected both, so an entirely correct extraction was
    # refused three times in a row and the sentence was abandoned with zero
    # facts. Tracking whether a candidate antecedent has already been seen
    # separates them: a pronoun preceded by a noun or proper noun in the same
    # claim can bind to it; one that appears before any noun cannot.
    #
    # This is a heuristic, not coreference resolution. It mishandles cataphora
    # ("Because IT was costly, the bridge was delayed") and cannot arbitrate
    # between two candidate antecedents. See the note in the module docstring
    # about promoting this to a real coref model.
    seen_candidate_antecedent = False

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            seen_candidate_antecedent = True
            continue

        if token.pos_ != "PRON":
            continue
        if token.tag_ in ("WDT", "WP", "WP$", "WRB"):   # relativiser
            continue

        lower = token.text.lower()
        if lower in _GENERIC_PRONOUNS:
            continue
        if lower in _ANAPHORIC and not seen_candidate_antecedent:
            return (
                f"'{token.text}' refers to something outside the claim — "
                f"replace it with the name it stands for, using wording from the document"
            )

    return None


def check_worthy(claim: str) -> str | None:
    """
    Is this claim worth sending to retrieval + verification?

    Returns None when check-worthy, otherwise the reason it is not.

    Rejects the two cases that cost ~152s to discover the expensive way and
    always come back NOT ENOUGH INFO — indistinguishable, in the final report,
    from a genuine retrieval failure:

      - Pure opinion: "The Bosporus is beautiful." No evidence can settle it.
      - No referent:  "It is very long." Nothing nameable to retrieve on.

    Conservative by design. A false reject silently drops a checkable claim,
    which is worse than paying 152s, so anything carrying a proper noun is
    admitted even if it also carries a subjective word — "Istanbul is the most
    populous city in Turkey" is both evaluative-sounding and perfectly
    verifiable.
    """
    doc = NLP(claim)

    has_proper_noun = any(t.pos_ == "PROPN" for t in doc)
    has_number      = any(t.like_num for t in doc)

    if has_proper_noun or has_number:
        return None

    # Match on BOTH the lemma and the surface form. The epistemic markers are
    # lemmas ('seem' catches 'seems', 'seemed'), but the Opinion Lexicon stores
    # surface forms, and lemmatising an adjective can move it off the entry —
    # so checking only one of the two would silently miss half the vocabulary.
    subjective = {
        t.lemma_.lower() for t in doc if t.lemma_.lower() in _SUBJECTIVE_LEMMAS
    } | {
        t.text.lower() for t in doc if t.text.lower() in _SUBJECTIVE_LEMMAS
    }
    if subjective:
        return (
            f"it expresses an opinion ({', '.join(sorted(subjective))}) about no "
            f"named entity — there is no evidence that could confirm or deny it"
        )

    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return "it names nothing concrete that could be looked up"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — Grammatical Subject Recovery
# ─────────────────────────────────────────────────────────────────────────────
_LEADING_ARTICLE = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)

# Dependency relations that hang extra CLAUSES or alternatives off a noun.
# They belong to the sentence, not to the entity's name, and dragging them into
# a search query destroys it: "the bridge that connects Asia and Europe" would
# otherwise be queried in full.
_SPAN_STOP_DEPS = {"relcl", "acl", "advcl", "appos", "conj", "cc", "punct"}

# Upper bound on subject length in tokens. A name longer than this is a parse
# artefact rather than an entity worth searching for.
_MAX_SUBJECT_TOKENS = 7


def _subject_span(head) -> str:
    """
    Text of the subject noun phrase, INCLUDING prepositional attachments.

    Written to replace a doc.noun_chunks lookup that silently truncated names.
    spaCy's noun chunks are *base* noun phrases and exclude PP attachments, so
    "The Republic of Sale" chunked as "The Republic" and the query was built
    from the bare word "Republic". That retrieved Spain in the 17th century and
    17th century BC, and the claim came back NOT ENOUGH INFO despite being
    stated verbatim in the Republic of Salé article.

    Walking token.subtree keeps "of Sale" attached, but the subtree also drags
    in relative clauses and coordinated alternatives — so anything reached
    through a relation in _SPAN_STOP_DEPS is pruned, and the result is the
    contiguous run of surviving tokens containing the head.
    """
    doc = head.doc

    # Compare by INDEX, never by identity. spaCy builds a new Token object on
    # every attribute access, so `cur is not head` is true even when both refer
    # to the same word — the ascent never terminates and the request hangs.
    # The ROOT guard is the second half of the same lesson: at the root,
    # token.head is the token itself, so without it the walk would spin there.
    keep = set()
    for tok in head.subtree:
        cur, pruned = tok, False
        while cur.i != head.i:
            if cur.dep_ in _SPAN_STOP_DEPS:
                pruned = True
                break
            if cur.head.i == cur.i:      # reached ROOT without passing head
                break
            cur = cur.head
        if not pruned:
            keep.add(tok.i)

    # Contiguity matters: a gap means the intervening words were pruned, and
    # splicing across it would invent a phrase the sentence never contained.
    start = end = head.i
    while start - 1 in keep:
        start -= 1
    while end + 1 in keep:
        end += 1

    # Over the limit, fall back to the HEAD NOUN rather than slicing the phrase.
    #
    # The cap used to truncate positionally, which produced query terms that
    # mean nothing: "The title of the world's longest river" is 8 tokens, so a
    # cap of 7 emitted "title of the world's longest" and the query anchored on
    # a dangling adjective. No value of the cap fixes that — 8 tokens fails at
    # 9 — because the bug is the behaviour at the boundary, not its position.
    #
    # "river" is a weak anchor, but it is a real noun phrase; "longest" is not.
    # Degrading to something smaller and correct beats something longer and
    # malformed, and it leaves the entity path free to supply a better anchor.
    if end - start + 1 > _MAX_SUBJECT_TOKENS:
        # The head of an OVERLONG subject phrase is generic by construction:
        # phrases get long precisely because the head is vague and needs
        # qualifying. "The title of the world's longest river" heads on
        # 'title'; "the city of the seven hills" on 'city'.
        #
        # Returning that head made retrieval WORSE than doing nothing —
        # promotion put the bare word 'title' ahead of 'the Amazon River' and
        # the retriever fetched an article about titles. Only a proper noun is
        # worth promoting; anything else yields None so the entity path takes
        # over cleanly.
        if head.pos_ == "PROPN":
            print(f"[NER] Subject phrase exceeds {_MAX_SUBJECT_TOKENS} tokens — "
                  f"falling back to the proper-noun head '{head.text}'.")
            return head.text

        print(f"[NER] Subject phrase exceeds {_MAX_SUBJECT_TOKENS} tokens and heads "
              f"on the common noun '{head.text}' — no subject promoted.")
        return ""

    return doc[start:end + 1].text


def extract_subject(fact: str) -> str | None:
    """
    Recover the grammatical subject of a claim via the dependency parse.

    WHY THIS EXISTS: NER and the query builder can both silently discard the
    entity the claim is actually about.

      Claim : "Bosporus is located between Africa and Europe."
      NER   : [('Africa', 'LOC'), ('Europe', 'LOC')]        <- no Bosporus
      Query : "Africa Europe"

    The Bosporus article was therefore never retrieved, the verifier saw only
    chunks about African settlement patterns, and it returned NOT ENOUGH INFO
    for a claim that Wikipedia flatly refutes. Two independent causes:
    en_core_web_sm often misses a bare proper noun with no determiner ("the
    Bosporus" is recognised far more reliably than "Bosporus"), and even when
    NER does fire, build_query() caps at the top-2 priority entities — where
    the subject can lose to its own objects, since LOC and GPE tie.

    The dependency parser is the right tool here because it is far more robust
    than the NER model for this particular question: identifying the nsubj of
    a clause needs only syntax, not knowledge of what the token refers to. It
    finds "Bosporus" whether or not any gazetteer has heard of it.

    Returns the subject noun phrase with any leading article stripped, or None
    for pronoun subjects ("It connects the Black Sea"), which make useless
    search queries and should fall through to the entity-based path.
    """
    doc = NLP(fact)

    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        if token.pos_ == "PRON":
            print(f"[NER] Subject '{token.text}' is a pronoun — not usable as a query anchor.")
            return None

        subject = _LEADING_ARTICLE.sub("", _subject_span(token)).strip()
        if subject:
            print(f"[NER] Grammatical subject: '{subject}'")
            return subject

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Query Construction
# ─────────────────────────────────────────────────────────────────────────────
# ── Tunable query-construction parameters ─────────────────────────────────────
# MAX_QUERY_ENTITIES was the literal `filtered_entities[:2]` inside
# build_query(). It is the single most consequential number in the retrieval
# path — it decides which entities survive into the search string — and it was
# invisible. Two is a genuine trade-off, not an obvious default:
#
#   fewer  → generic queries that return huge topic articles
#            ('Africa Europe' retrieved a migration-patterns passage)
#   more   → over-specification; no article contains every term
#
# It interacts with ENTITY_PRIORITY: the cap only matters because entities are
# ranked, and a mis-ranked entity beyond the cap is discarded entirely. That is
# how 'Yavuz Sultan Selim Bridge' (ORG, 3) lost to 'Africa'/'Europe' (LOC, 4)
# and never reached the index.
#
# Raised from 2 to 3 because EXCLUSION was the only harm ranking ever did.
# extract_queries() already issues a standalone query per surviving entity, so
# a mis-ranked entity that stays inside the cap costs nothing — it still gets
# its own lookup. An entity pushed outside the cap is never searched for at
# all, which is unrecoverable. Widening it converts a ranking problem into an
# ordering preference, at the price of one extra Wikipedia call (~5-15s against
# ~190s verifications).
MAX_QUERY_ENTITIES = 3


def _specificity(entity: str) -> tuple[int, int, int]:
    """
    Sort key ranking an entity string by how narrowly it identifies an article.

    Replaces ENTITY_PRIORITY's role in ORDERING (the table still gates DATE
    entities in filter_entities). The type table asserted a total order over
    categories — PERSON > FAC > LOC > ORG — that does not exist: a person is
    not inherently a better search anchor than a place. What actually predicts
    a good anchor is SPECIFICITY, which is a property of the individual string,
    not of its predicted category.

    Worked example, "Europe connects to Asia by a land bridge called Yavuz
    Sultan Selim Bridge":

        entity                     by type   by grammatical role   here
        Europe                     1st       1st (subject)         3rd
        Asia                       2nd       2nd (object)          2nd
        Yavuz Sultan Selim Bridge  3rd       3rd (oblique)         1st

    Only the last column is right — and note that grammatical role gets it
    exactly backwards here, because the specific entity sits in a prepositional
    phrase while two continents occupy subject and object. Role answers "what
    is this claim about?" (which is why subject promotion still runs first, and
    why diagnose_nei uses it); it does not answer "what is a good search term?"

    Signals, in order of the tuple, all descending:
      1. token count      — multi-token names identify fewer articles
      2. title-cased      — a proper name rather than a common noun
      3. character length — a weak tiebreak, longer strings are rarer

    Deliberately no corpus frequency: that would need an index or a download,
    and these three are computable from the string alone.
    """
    tokens = entity.split()
    return (
        len(tokens),
        sum(1 for t in tokens if t[:1].isupper()),
        len(entity),
    )


def _by_specificity(entities: list[str]) -> list[str]:
    """Most specific first. Stable, so equal keys keep ENTITY_PRIORITY order."""
    return sorted(entities, key=_specificity, reverse=True)


def _sanitize_query(text: str) -> str:
    """
    Shared query sanitizer used by build_query() and extract_queries().

    - Strips English possessive suffixes BEFORE deleting apostrophes —
      otherwise "Gustave Eiffel's" collapses into the garbage token
      "Gustave Eiffels" and the Wikipedia search misses the article.
      Covers both the straight apostrophe and the typographic U+2019.
    - Removes syntax characters that invalidate search engine parsers.
    - Collapses whitespace.
    """
    text = re.sub(r"[’']s\b", "", text)
    text = re.sub(r'[\"\'(){}\[\]]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_query(filtered_entities: list[str], fact: str) -> str | None:
    """
    Synthesizes prioritized entities into a clean, targeted query string.

    Strategy for Information Density Optimization:
    - 1 entity  -> Extracted directly as the primary anchor.
    - 2 entities -> Combined sequentially to form a precise multi-hop intersection.
    - 3+ entities -> Caps extraction at the top 2 highest priority entities. This 
                     prevents over-specification, which causes document retrieval 
                     misses in standard indexing engines.
    """
    if not filtered_entities:
        return None

    top_entities = filtered_entities[:MAX_QUERY_ENTITIES]

    query = _sanitize_query(" ".join(top_entities))

    print(f"[NER] Built query: '{query}'")
    return query if query else None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Fallback Keyword Extraction
# ─────────────────────────────────────────────────────────────────────────────
def fallback_keyword_extraction(fact: str) -> str:
    """
    Fallback mechanism triggered when spaCy fails to identify explicit named entities.
    Strips grammatical punctuation and language stopwords, extracting raw, 
    low-level descriptive content tokens.

    Example:
        Fact: "Large language models mitigate structural hallucinations."
        Filtered Keywords: ["language", "models", "mitigate", "hallucinations"]
    """
    # Isolate atomic alphabetic terms, casting to lowercase to align with index tokens
    tokens = re.findall(r'\b[a-zA-Z]+\b', fact.lower())

    # Filter language noise and drop shorthand words
    keywords = [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]

    # Constrain the search phrase to the first 4 high-density concepts to maintain focus
    query = " ".join(keywords[:4])

    print(f"[NER] Fallback query generated: '{query}'")
    return query


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# ─────────────────────────────────────────────────────────────────────────────
def extract_query(fact: str) -> str:
    """
    Executes the comprehensive Named Entity Query Extraction pipeline.
    
    Execution Flow:
    [Atomic Fact input] -> extract_entities() -> filter_entities() -> build_query()
                                                                          |
                                                      (If Query is Empty) v
                                                           fallback_keyword_extraction()
    """
    print(f"\n[NER] Processing atomic unit: '{fact}'")

    entities = extract_entities(fact)
    filtered = filter_entities(entities)
    query    = build_query(filtered, fact)

    # Trigger fallback logic if entity structures yield an empty sequence
    if query is None:
        print("[NER] Structural resolution failed. Triggering semantic keyword fallback extraction.")
        query = fallback_keyword_extraction(fact)

    print(f"[NER] Final synthesized query: '{query}'")
    return query


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Multi-Query Extraction (used by retriever.fetch)
# ─────────────────────────────────────────────────────────────────────────────
def extract_queries(fact: str, max_queries: int = MAX_QUERY_ENTITIES + 1) -> list[str]:
    """
    Multi-query variant of extract_query() for robust retrieval.

    Single-query retrieval has a brittle failure mode: the query anchors on
    the top priority entities, so ONE hallucinated or mis-typed entity
    poisons the entire evidence pool for the fact. Observed examples:
    "Gustave Eiffel Bedford Basin" retrieved a World Heritage Sites list,
    and spaCy tagging "Statue of Liberty" as ORG dropped it from the top-2
    cut entirely. Issuing an additional standalone query per top entity
    means at least one query still lands on the right article.

    ORDERING IS NOW TWO-STAGE, and the two stages answer different questions:

        subject promotion  — "what is this claim ABOUT?"      (grammatical role)
        _by_specificity    — "what is a good SEARCH TERM?"    (string properties)

    The subject goes first because a claim's subject must always be searched
    for. Everything after it is ordered by specificity rather than by
    ENTITY_PRIORITY, because entity TYPE predicts neither question well —
    'Yavuz Sultan Selim Bridge' is tagged ORG and outranked by the continents
    it connects, though it names exactly one article and they name vast ones.

    The default max_queries tracks MAX_QUERY_ENTITIES so that every entity
    surviving the cap gets its own standalone lookup; otherwise widening the
    cap would admit an entity to the combined query while still denying it a
    query of its own.

    Returns (deduplicated, priority order):
        1. The combined query over the top MAX_QUERY_ENTITIES entities
        2. One standalone query per surviving entity — date entities excluded,
           since a bare "1889" retrieves the year article (pure noise)
        3. Keyword fallback if nothing else was produced
    """
    print(f"\n[NER] Processing atomic unit (multi-query): '{fact}'")

    entities = extract_entities(fact)
    filtered = filter_entities(entities)

    # SUBJECT PROMOTION: whatever the claim is *about* must reach the index.
    #
    # The subject is MOVED to the front, not merely inserted when absent. The
    # earlier version only prepended a subject that NER had missed entirely,
    # which left the case that actually matters unfixed: NER finds the entity,
    # but ENTITY_PRIORITY ranks it below the claim's objects, so build_query()'s
    # top-2 cap discards it anyway. Observed exactly that on
    #
    #   "Yavuz Sultan Selim Bridge was located between Africa and Europe."
    #   filtered -> ['Africa', 'Europe', 'Yavuz Sultan Selim Bridge']   (LOC 4 > ORG 3)
    #   query    -> 'Africa Europe'                                     (bridge cut)
    #
    # Rebuilding the list rather than inserting also de-duplicates: an entity
    # matching the subject by substring either way ('Bosporus' vs 'the
    # Bosporus') is dropped from the tail so it cannot occupy a second slot.
    subject = extract_subject(fact)
    if subject:
        # UPGRADE BEFORE PROMOTING. If NER recognised a fuller form of the same
        # entity, use NER's string. Without this, a truncated subject would
        # evict the better name during de-duplication: the subject 'Republic'
        # matched the entity 'The Republic of Sale' by substring, so the entity
        # was dropped as a duplicate and the query was built from the fragment.
        # Promotion made retrieval worse than leaving it alone.
        for entity in filtered:
            if subject.lower() in entity.lower() and len(entity) > len(subject):
                print(f"[NER] Subject '{subject}' upgraded to NER's fuller form '{entity}'.")
                subject = entity
                break

        rest = [
            e for e in filtered
            if subject.lower() not in e.lower() and e.lower() not in subject.lower()
        ]
        if filtered[:1] != [subject]:
            print(f"[NER] Promoting subject '{subject}' to the head of the query.")
        filtered = [subject] + _by_specificity(rest)
    else:
        filtered = _by_specificity(filtered)

    queries: list[str] = []

    combined = build_query(filtered, fact)
    if combined:
        queries.append(combined)

    for ent_text in filtered[:MAX_QUERY_ENTITIES]:
        if len(queries) >= max_queries:
            break
        q = _sanitize_query(ent_text)
        # Skip empty strings, standalone date-like entities, and duplicates
        if not q or re.fullmatch(r'[\d\s\-–to]+', q) or q in queries:
            continue
        queries.append(q)

    if not queries:
        print("[NER] Structural resolution failed. Triggering semantic keyword fallback extraction.")
        queries.append(fallback_keyword_extraction(fact))

    print(f"[NER] Final query set: {queries}")
    return queries