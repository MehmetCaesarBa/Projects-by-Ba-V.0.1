"""
stemmer_data.py — evaluation sets for stemmer_bench.py.

Separated from the benchmark logic so the data can grow without the scoring
code becoming unreadable, and so the sets can be reused by other experiments.

Four sets, each measuring a different failure the faithfulness gate can suffer:

    A  MUST_COLLAPSE      inflections that must map together   -> misses cause FALSE REJECTIONS
    B  MUST_NOT_COLLAPSE  distinct words that must stay apart  -> misses cause FALSE ACCEPTANCES
    C  PROPER_NOUNS       names that must survive intact       -> fact-checking is full of them
    D  GATE_CASES         real (document, claim) pairs         -> the decisive tier
"""

# ═════════════════════════════════════════════════════════════════════════════
# SET A — must collapse (50)
#
# A stemmer that fails these rejects faithful claims: the extractor rewrites a
# clause into a standalone sentence and the inflection shifts, so 'separated'
# has to reach the same key as 'separates' or it looks like invented material.
#
# The last block is IRREGULAR morphology, which no suffix-stripping algorithm
# can reach — 'went' shares no letters with 'go'. Only a lemmatiser passes
# those, so they are the sharpest discriminator in the set. They are kept to
# eight of fifty so they inform the result without dominating it.
# ═════════════════════════════════════════════════════════════════════════════
MUST_COLLAPSE = [
    # regular verb inflection (16)
    ("separates", "separated"),   ("connects", "connecting"),
    ("located", "locate"),        ("produced", "produce"),
    ("discovered", "discover"),   ("established", "establish"),
    ("published", "publish"),     ("defeated", "defeat"),
    ("designed", "design"),       ("operated", "operate"),
    ("governed", "govern"),       ("invaded", "invade"),
    ("occupied", "occupy"),       ("translated", "translate"),
    ("constructed", "construct"), ("completed", "complete"),

    # noun plurals (16)
    ("prizes", "prize"),          ("cities", "city"),
    ("countries", "country"),     ("boundaries", "boundary"),
    ("families", "family"),       ("studies", "study"),
    ("armies", "army"),           ("treaties", "treaty"),
    ("territories", "territory"), ("universities", "university"),
    ("libraries", "library"),     ("centuries", "century"),
    ("bridges", "bridge"),        ("mountains", "mountain"),
    ("rivers", "river"),          ("islands", "island"),

    # comparatives, superlatives, adverbs (5)
    ("larger", "large"),          ("largest", "large"),
    ("longest", "long"),          ("quickly", "quick"),
    ("historically", "historical"),

    # progressive forms (5)
    ("building", "build"),        ("running", "run"),
    ("swimming", "swim"),         ("writing", "write"),
    ("beginning", "begin"),

    # irregular — lemmatiser only (8)
    ("went", "go"),               ("wrote", "write"),
    ("built", "build"),           ("took", "take"),
    ("gave", "give"),             ("children", "child"),
    ("men", "man"),               ("feet", "foot"),
]

# ═════════════════════════════════════════════════════════════════════════════
# SET B — must NOT collapse (50)
#
# Weighted toward the endings stemmers actually strip (-ity, -ation, -ive,
# -or, -al, -ism), because pairs no algorithm would ever conflate make the
# score look good without testing anything. The first block is the classic
# over-stemming list from the IR literature.
#
# A stemmer that fails these accepts unfaithful claims: an invented word
# collapses into a word the source happens to contain, and the residue that
# should have flagged it disappears.
# ═════════════════════════════════════════════════════════════════════════════
MUST_NOT_COLLAPSE = [
    # classic conflation errors (14)
    ("university", "universe"),   ("organization", "organ"),
    ("policy", "police"),         ("general", "generation"),
    ("operate", "operator"),      ("relative", "relate"),
    ("nation", "nature"),         ("communism", "community"),
    ("execute", "executive"),     ("numerical", "numerous"),
    ("probable", "probe"),        ("experiment", "experience"),
    ("organic", "organ"),         ("national", "native"),

    # geography and history vocabulary — this project's actual domain (12)
    ("continental", "continent"), ("mountain", "mount"),
    ("kingdom", "king"),          ("frontier", "front"),
    ("island", "isle"),           ("population", "popular"),
    ("colonial", "colony"),       ("imperial", "empire"),
    ("regional", "region"),       ("province", "prove"),
    ("territory", "terror"),      ("civilian", "civil"),

    # short stems that aggressive algorithms crush together (12)
    ("arm", "army"),              ("news", "new"),
    ("plan", "planet"),           ("cell", "cellar"),
    ("car", "care"),              ("fund", "fun"),
    ("single", "sing"),           ("rate", "rat"),
    ("mineral", "mine"),          ("business", "bus"),
    ("season", "sea"),            ("article", "art"),

    # -ion / -ment / -ary endings (12)
    ("mission", "missionary"),    ("statement", "state"),
    ("department", "depart"),     ("document", "dock"),
    ("monument", "monk"),         ("station", "state"),
    ("commission", "commit"),     ("addition", "add"),
    ("edition", "edit"),          ("division", "divide"),
    ("tradition", "trade"),       ("solution", "solve"),
]

# ═════════════════════════════════════════════════════════════════════════════
# SET C — proper nouns, IN CONTEXT (50)
#
# Each entry is (word, sentence). The sentence exists because the first run of
# this benchmark could not distinguish spacy_lemma from spacy_lemma+propn_skip:
# given the bare token "Wales", spaCy has no way to know it is a country rather
# than the plural of 'wale' (a ridge in fabric), so the PROPN branch never
# fired and the two candidates scored identically.
#
# One uniform template is used rather than fifty hand-written sentences, so no
# candidate benefits from a sentence that happens to suit it. Subject position
# plus capitalisation is the strongest available proper-noun signal.
#
# Names ending in -s dominate deliberately: they are the failure mode. Snowball
# leaves 'Bosporus' alone (Porter2 exempts words ending in 'us') but mangles
# 'Paris' to 'pari' and 'Athens' to 'athen'.
# ═════════════════════════════════════════════════════════════════════════════
_TEMPLATE = "{} is described in the encyclopedia entry."

_PROPER_NOUN_WORDS = [
    # straits, mountains, rivers (12)
    "Bosporus", "Elbrus", "Ararat", "Everest", "Kilimanjaro", "Vesuvius",
    "Thames", "Ganges", "Indus", "Euphrates", "Tigris", "Andes",
    # cities (12)
    "Paris", "Athens", "Naples", "Brussels", "Marseilles", "Istanbul",
    "Hornachos", "Damascus", "Lagos", "Buenos", "Cannes", "Rhodes",
    # countries and regions (13)
    "Wales", "Texas", "Kansas", "Illinois", "Arkansas", "Massachusetts",
    "Netherlands", "Philippines", "Maldives", "Bahamas", "Honduras",
    "Cyprus", "Belarus",
    # people (13)
    "Socrates", "Archimedes", "Descartes", "Cervantes", "Dickens", "Keynes",
    "Hobbes", "Marx", "Engels", "Curie", "Copernicus", "Erasmus", "Aristotle",
]

PROPER_NOUNS = [(w.lower(), _TEMPLATE.format(w)) for w in _PROPER_NOUN_WORDS]

# ═════════════════════════════════════════════════════════════════════════════
# SET D — real gate decisions
#
# Sets A-C measure linguistic behaviour. This measures whether the GATE reaches
# the right answer, which is the only thing that matters. Cases are harvested
# from actual pipeline runs plus constructed variants that isolate one
# behaviour each.
#
# expected_foreign: {} for a faithful extraction, otherwise the invented words.
# ═════════════════════════════════════════════════════════════════════════════
GATE_CASES = [
    {
        "note": "faithful — the falsehood is preserved",
        "document": "Bosporus is located between Africa and Europe.",
        "claim": "Bosporus is located between Africa and Europe.",
        "expected_foreign": set(),
    },
    {
        "note": "the original bug — model substituted Asia for Africa",
        "document": "Bosporus is located between Africa and Europe.",
        "claim": "The Bosporus is located between Asia and Europe.",
        "expected_foreign": {"asia"},
    },
    {
        "note": "inflection only — 'separated' vs 'separates'",
        "document": "The Bosporus separates Africa and Europe.",
        "claim": "Africa and Europe are separated by the Bosporus.",
        "expected_foreign": set(),
    },
    {
        "note": "plural shift — 'boundaries' vs 'boundary'",
        "document": "The strait forms one of the continental boundaries.",
        "claim": "The strait forms a continental boundary.",
        "expected_foreign": set(),
    },
    {
        "note": "world knowledge added from memory",
        "document": "The Republic of Sale traces its origins to the 17th century.",
        "claim": "The Republic of Sale was established in the 17th century.",
        "expected_foreign": {"establish"},
    },
    {
        "note": "faithful, long entity names intact",
        "document": ("Yavuz Sultan Selim Bridge was located between Africa and "
                     "Europe but it was destroyed by British dreadnoughts."),
        "claim": "Yavuz Sultan Selim Bridge was destroyed by British dreadnoughts.",
        "expected_foreign": set(),
    },
    {
        "note": "verifier's correction recycled as a claim",
        "document": "Bosporus is located between Africa and Europe.",
        "claim": "The Bosporus forms the continental boundary between Asia and Europe.",
        "expected_foreign": {"asia", "continental", "boundary", "form"},
    },
    {
        "note": "proper noun must survive — Paris is not a plural",
        "document": "The Eiffel Tower is located in Paris.",
        "claim": "The Eiffel Tower is located in Paris.",
        "expected_foreign": set(),
    },
    {
        "note": "proper noun must survive — Athens",
        "document": "The Parthenon stands in Athens.",
        "claim": "The Parthenon is located in Athens.",
        "expected_foreign": set(),
    },
    {
        "note": "over-stemming trap — 'organization' must not match 'organ'",
        "document": "The organ was installed in the cathedral in 1750.",
        "claim": "The organization was founded in the cathedral in 1750.",
        "expected_foreign": {"organization", "found"},
    },
    {
        "note": "over-stemming trap — 'university' must not match 'universe'",
        "document": "The universe is expanding at an accelerating rate.",
        "claim": "The university is expanding at an accelerating rate.",
        "expected_foreign": {"university"},
    },
    {
        "note": "over-stemming trap — 'mountain' must not match 'mount'",
        "document": "Mount Elbrus rises above the Caucasus range.",
        "claim": "The mountain rises above the Caucasus range.",
        "expected_foreign": {"mountain"},
    },
    {
        "note": "irregular verb — 'wrote' vs 'writes'",
        "document": "Cervantes writes about a knight from La Mancha.",
        "claim": "Cervantes wrote about a knight from La Mancha.",
        "expected_foreign": set(),
    },
    {
        "note": "irregular plural — 'children' vs 'child'",
        "document": "The child was educated in Vienna.",
        "claim": "The children were educated in Vienna.",
        "expected_foreign": set(),
    },
    {
        "note": "fabricated date",
        "document": "Jamestown was founded in May 1607.",
        "claim": "Jamestown was founded in September 1607.",
        "expected_foreign": {"september"},
    },
    {
        "note": "fabricated entity",
        "document": "The Manhattan Project produced the first nuclear weapons.",
        "claim": "The Manhattan Project produced the first hydrogen bombs.",
        "expected_foreign": {"hydrogen", "bomb"},
    },
    {
        "note": "NEGATION FLIP — known blind spot, excluded from scoring",
        "document": "It is universally acknowledged as longer than the Nile.",
        "claim": "The Nile is not universally acknowledged as longer.",
        "expected_foreign": {"<negation>"},
    },
]
