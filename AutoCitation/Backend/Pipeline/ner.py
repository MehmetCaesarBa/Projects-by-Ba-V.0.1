import re
import spacy

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
    
    # 4. GPE: Geopolitical entities (countries, cities, sovereign states). 
    #    Provides an immediate geographical constraint to eliminate out-of-domain noise.
    "GPE"    : 4,
    
    # 5. EVENT: Named battles, tech conferences, or historic occurrences (e.g., "WWII", "ROSCon").
    #    Strongly binds specific timeline occurrences together.
    "EVENT"  : 5,
    
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

    entity_types  = {e[1] for e in sorted_entities}
    
    # Check if there is an active structural anchor present alongside a date
    has_high_prio = any(
        ENTITY_PRIORITY.get(t, 0) >= ENTITY_PRIORITY["NORP"]
        for t in entity_types
    )

    filtered = []
    for text, label in sorted_entities:
        # Prevent query pollution by filtering out isolated chronological variables
        if label == "DATE" and not has_high_prio:
            print(f"[NER] Dropping isolated date entity: '{text}'")
            continue
        filtered.append(text)

    print(f"[NER] Filtered entities: {filtered}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Query Construction
# ─────────────────────────────────────────────────────────────────────────────
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

    # Cap at top 2 entities to maintain search breadth while ensuring query precision
    top_entities = filtered_entities[:2]

    query = " ".join(top_entities)

    # Sanitize query parameters to strip syntax that invalidates search engine parsers
    query = re.sub(r'[\"\'(){}\[\]]+', '', query)
    query = re.sub(r'\s+', ' ', query).strip()

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