"""
program.py — program-guided verification for claims the single-shot verifier
cannot decide.

INSPIRED BY, NOT FAITHFUL TO, ProgramFC (Pan et al., ACL 2023,
arXiv:2305.12744). The paper generates reasoning programs with an LLM using
in-context demonstrations. Here the programs are SYNTHESISED FROM THE PARSE,
deterministically, with no extra inference. That is a deliberate divergence and
the thesis should describe it as such.

    the paper's way   general, handles claims nobody anticipated, needs a
                      capable LLM to emit valid programs, one extra call per
                      claim (30-120s on this hardware), and a parser plus a
                      fallback for malformed output.

    this way          covers exactly the operator classes that appear in the
                      failure logs, costs no inference, and is unit-testable
                      offline against a fixed parse.

WHY A PROGRAM AT ALL. The single-shot verifier is asked "does this evidence
support this claim?" — one question, one answer. For two claim types that
question is the wrong one, and no prompt wording repairs it:

  SUPERLATIVE   "Jamestown is the earliest European settlement in the US."
                Evidence about Jamestown cannot refute this, because the
                refutation is a passage about SAINT AUGUSTINE, which does not
                contain the word 'Jamestown' and therefore never ranks. Three
                verifier prompt edits produced three different verdicts on
                identical evidence — the signature of an under-determined
                question rather than a wording problem.

                The program asks a different question: WHO ACTUALLY WAS FIRST?
                Then it compares that answer to the claim's subject. Retrieval
                is aimed at the category, not the subject, so the evidence that
                settles it is reachable.

  COMPARATIVE   "Mount Everest is taller than K2."
                Needs two numbers from two articles and an arithmetic
                comparison. The pipeline has no stage where two facts meet, and
                an 8B model comparing 8849 against 8611 is unreliable because a
                tokenizer splits numbers into arbitrary pieces. The program
                retrieves each side, extracts the quantity, and compares in
                Python — where comparison is exact and the working is citable.

WHAT IT DELIBERATELY DOES NOT DO. Return None for anything it does not
recognise. Most claims carry no operator, the existing path handles them, and a
program that fires on everything would be a rewrite of the verifier rather than
an addition to it. Every uncertain branch resolves to NOT ENOUGH INFO rather
than to a guess: this module can only ever move a verdict when it is confident,
which keeps its failure mode "no help" instead of "new wrong answers".
"""

import re
from dataclasses import dataclass, field

import Pipeline.ner as ner
import Pipeline.retriever as retriever

import requests

# ── Config ────────────────────────────────────────────────────────────────────
# Master switch. False restores single-shot verification for every claim, which
# is the comparison baseline — the claim "programs beat single-shot on
# superlatives" should be reproducible, not asserted.
PROGRAM_MODE = True

OLLAMA_URL = "http://localhost:11434/api/generate"
ANSWER_MODEL = "phi3:mini"
KEEP_ALIVE = "30m"

# The Question() handler is EXTRACTIVE, not evaluative: "which name does this
# passage give for X?" is a copying task, so the fast model is the right tool
# and the reasoning model would be paying 100-600s for a job that needs none.
#
# The safety net is not the model's care, it is the containment check in
# _answer_is_grounded: an answer whose words do not appear in the evidence is
# discarded. That makes hallucination a deterministic rejection rather than a
# probabilistic hope.
ANSWER_NUM_PREDICT = 48

# How many chunks the Question handler reads. Higher than the verifier's budget
# because this is a lookup over a category rather than a judgement about one
# sentence — the answer may sit outside the top two.
QUESTION_TOP_K = 4


# ── Operator detection ────────────────────────────────────────────────────────
# Comparative markers. JJR/RBR are the comparative tags ('taller', 'longer',
# 'more'). A comparative alone is not enough — "the river is longer now than in
# 1970" compares one thing to itself across time — so a comparison also requires
# a 'than' with a nominal object to compare AGAINST.
_COMPARATIVE_TAGS = {"JJR", "RBR"}

# Dimension lexicon: which quantity a comparative adjective is about, and the
# family of units that expresses it. Without this, "taller" and "longer" are
# indistinguishable strings and the extractor cannot tell which of an article's
# many numbers is the relevant one.
#
# Deliberately small. Each entry is a claim about what a word measures, and a
# wrong entry produces a confident comparison of unrelated quantities — far
# worse than declining. Ambiguous adjectives ('larger', 'bigger', 'greater')
# are EXCLUDED for exactly that reason: larger by area, population, or volume
# is not decidable from the adjective.
_DIMENSIONS = {
    "tall":     ("height", "length"),
    "high":     ("height", "length"),
    "long":     ("length", "length"),
    "deep":     ("depth", "length"),
    "wide":     ("width", "length"),
    "populous": ("population", "count"),
    "old":      ("age", "year"),
}

# Unit conversion to a canonical base. Length -> metres, count -> units,
# year -> year. Only units that actually appear in encyclopedia prose.
_UNIT_TO_BASE = {
    # length family -> metres
    "mm": 0.001, "cm": 0.01, "m": 1.0, "metre": 1.0, "metres": 1.0,
    "meter": 1.0, "meters": 1.0, "km": 1000.0, "kilometre": 1000.0,
    "kilometres": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}

_UNIT_FAMILY = {u: "length" for u in _UNIT_TO_BASE}


# ── Program representation ────────────────────────────────────────────────────
@dataclass(frozen=True)
class Step:
    """One instruction. `op` names the handler, `args` are its inputs."""
    var: str
    op: str
    args: tuple


@dataclass
class Program:
    """
    A synthesised reasoning program plus the bindings produced by executing it.

    `trace` is not decoration. ProgramFC's stated advantage over end-to-end
    verification is that the reasoning is inspectable and debuggable, and a
    program whose intermediate answers are invisible gives that up. The trace
    becomes part of the rationale shown to the user, so a verdict can be argued
    with rather than merely accepted.
    """
    kind: str
    steps: list[Step]
    bindings: dict = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    source_url: str = ""
    query: str = ""


# ── Synthesis ─────────────────────────────────────────────────────────────────
def synthesize(claim: str) -> Program | None:
    """
    Build a program for `claim`, or None when no operator is recognised.

    Superlative is tested BEFORE comparative. "The Amazon is the longest river"
    contains 'longest' (JJS) and would also satisfy a loose comparative test;
    the superlative reading is the correct one, and running the comparative
    handler on it would look for a second entity that does not exist.
    """
    doc = ner.NLP(claim)

    superlative = _find_superlative(doc)
    if superlative is not None:
        return _superlative_program(claim, doc, superlative)

    comparative = _find_comparative(doc)
    if comparative is not None:
        return _comparative_program(claim, doc, *comparative)

    return None


def _find_superlative(doc):
    """The superlative or ordinal token, if the claim has one."""
    for token in doc:
        if token.tag_ in ner._SUPERLATIVE_TAGS:
            return token
        if token.lemma_.lower() in ner._SUPERLATIVE_LEMMAS:
            return token
    return None


def _find_comparative(doc):
    """
    (comparative_token, compared_entity) — or None.

    Requires BOTH a comparative tag and a 'than' phrase with a nominal object.
    "longer than the Nile" qualifies; "the river is longer now" does not, and
    treating it as a comparison would send the extractor looking for a second
    entity the claim never names.
    """
    comp = next((t for t in doc if t.tag_ in _COMPARATIVE_TAGS), None)
    if comp is None:
        return None

    than = next((t for t in doc if t.text.lower() == "than"), None)
    if than is None:
        return None

    # The compared entity is the nominal governed by 'than'. Taking its whole
    # subtree keeps multi-word names intact ('the Nile', 'Mount Kilimanjaro').
    for token in doc:
        if token.head.i == than.i or (token.i > than.i and token.pos_ in ("PROPN", "NOUN")):
            span = _noun_span(token)
            if span:
                return comp, span
    return None


def _noun_span(token) -> str:
    """Contiguous noun phrase around `token`, articles stripped."""
    doc = token.doc
    idx = sorted(
        t.i for t in token.subtree
        if t.pos_ in ("PROPN", "NOUN", "ADJ", "NUM") and not t.is_stop
    )
    if not idx:
        return ""
    text = doc[idx[0]: idx[-1] + 1].text
    return re.sub(r'^(the|a|an)\s+', '', text, flags=re.I).strip()


def _superlative_program(claim: str, doc, superlative) -> Program | None:
    """
    answer_1 = Question(<category query>)
    label    = Match(subject, answer_1)

    The whole point is in the first line: the query names the CATEGORY the
    superlative ranks within, not the claim's subject. Asking "what was the
    earliest European settlement in the US?" reaches Saint Augustine; asking
    about Jamestown never can.
    """
    subject = ner.extract_subject(claim)
    category = ner.build_predicate_query(claim)

    # Both are required. Without a subject there is nothing to compare the
    # answer against; without a category query the retrieval would fall back to
    # the subject and reproduce the failure this exists to fix.
    if not subject or not category:
        return None

    return Program(
        kind="superlative",
        steps=[
            Step("answer_1", "Question", (category,)),
            Step("label", "Match", ("answer_1", subject)),
        ],
        query=category,
    )


def _comparative_program(claim: str, doc, comp, other: str) -> Program | None:
    """
    q_1   = Quantity(subject, dimension)
    q_2   = Quantity(other,   dimension)
    label = Compare(q_1, q_2)

    Arithmetic happens in Python. A tokenizer splits '8849' into pieces that
    carry no magnitude, so asking an 8B model whether 8849 exceeds 8611 is a
    text-pattern question wearing a number's clothes. Python does not have that
    problem, and the two source values can be shown in the rationale.
    """
    subject = ner.extract_subject(claim)
    if not subject:
        return None

    dimension = _DIMENSIONS.get(comp.lemma_.lower())
    if dimension is None:
        # An unmapped comparative ('larger', 'better', 'more important') names
        # no measurable dimension, or an ambiguous one. Decline rather than
        # compare whatever numbers happen to be nearby.
        return None

    name, family = dimension
    return Program(
        kind="comparative",
        steps=[
            Step("q_1", "Quantity", (subject, name, family)),
            Step("q_2", "Quantity", (other, name, family)),
            Step("label", "Compare", ("q_1", "q_2")),
        ],
        query=f"{subject} {other}",
    )


# ── Handlers ──────────────────────────────────────────────────────────────────
def _retrieve(query: str, top_k: int) -> tuple[list[str], str]:
    """Chunks for a free-text query, ranked by the retriever's own scorer."""
    articles = retriever.fetch_articles(query)
    if not articles:
        return [], ""
    chunks = retriever.chunk_articles(articles)
    scored = retriever.score_chunks(query, chunks)
    if not scored:
        return [], ""
    return [c for _, c, _ in scored[:top_k]], scored[0][2]


def _log_stats(body: dict, tag: str) -> None:
    """
    Timing line for one inference, in the same shape the rest of the pipeline
    prints.

    WRITTEN LOCALLY RATHER THAN IMPORTED. The first version called
    ollama_client.log_inference_stats, which exists in some versions of this
    project and not others — a revert removed it, and the resulting
    AttributeError took down every program run until the fallback caught it.
    The module's whole value is being the path that works when the verifier
    cannot decide, so it should not depend on a helper for a print statement.
    Ten lines of duplication is the cheaper side of that trade.
    """
    ns = body.get("eval_duration") or 0
    tokens = body.get("eval_count") or 0
    seconds = ns / 1e9
    rate = tokens / seconds if seconds else 0.0
    print(f"[Inference] {tag:12} output {tokens:5d} tok in {seconds:6.1f}s ({rate:.1f} tok/s)")


def _call_answer_model(prompt: str) -> str:
    payload = {
        "model": ANSWER_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": 4096,
            "num_predict": ANSWER_NUM_PREDICT,
            "temperature": 0, "top_p": 1, "top_k": 1,
            "repeat_penalty": 1.0, "seed": 0,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    body = response.json()
    _log_stats(body, ANSWER_MODEL)
    return body["response"].strip()


def _answer_is_grounded(answer: str, chunks: list[str]) -> bool:
    """
    Does every content word of the answer appear in the evidence?

    THE ONLY THING STANDING BETWEEN THIS MODULE AND A FABRICATED VERDICT. The
    Question handler runs on phi3:mini, which knows a great deal about early
    American settlements and would happily answer from memory. An answer drawn
    from weights rather than from the retrieved text would then be compared
    against the claim's subject and could produce a confident REFUTES with a
    citation to a passage that never said it.

    Containment makes that a deterministic rejection. Same principle as the
    extractor's faithfulness gate, same reason.
    """
    haystack = " ".join(chunks).lower()
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) >= 4]
    if not words:
        return False
    return all(w in haystack for w in words)


def _handle_question(query: str) -> tuple[str | None, list[str], str]:
    """
    Question(query) -> the entity the evidence names, or None.

    Returns None rather than guessing whenever the evidence does not clearly
    answer, because None becomes NOT ENOUGH INFO downstream and a wrong name
    becomes a wrong verdict.
    """
    chunks, url = _retrieve(query, QUESTION_TOP_K)
    if not chunks:
        print(f"[Program] Question('{query}') — no evidence retrieved.")
        return None, [], ""

    evidence = "\n".join(f"Evidence_{i}: {c}" for i, c in enumerate(chunks, 1))
    prompt = f"""Read the evidence and answer the question with a NAME ONLY.

{evidence}

Question: What is the {query}?

Rules:
- Answer with the name of the place, person or thing, and nothing else.
- Copy the name exactly as it appears in the evidence.
- If the evidence does not clearly answer the question, reply exactly: UNKNOWN

Answer:"""

    raw = _call_answer_model(prompt)
    answer = raw.splitlines()[0].strip().strip('."') if raw else ""

    if not answer or answer.upper().startswith("UNKNOWN"):
        print(f"[Program] Question('{query}') -> UNKNOWN")
        return None, chunks, url

    if not _answer_is_grounded(answer, chunks):
        print(f"[Program] Question('{query}') -> '{answer}' REJECTED — not in the "
              f"evidence, so it came from the model's own knowledge.")
        return None, chunks, url

    if not _answer_is_right_kind(answer, query, chunks):
        return None, chunks, url

    print(f"[Program] Question('{query}') -> '{answer}'")
    return answer, chunks, url


def _answer_is_right_kind(answer: str, query: str, chunks: list[str]) -> bool:
    """
    Is the answer the same KIND of thing the question asked about?

    THE FAILURE THIS CATCHES. Asked "what was the earliest European permanent
    SETTLEMENT in the United States?", the model answered **'Juan Ponce de
    León'** — an explorer. Grounding passed, because the name does appear in the
    evidence. Match then compared a person against a settlement, found they
    differed, and returned REFUTES.

    The verdict happened to be right. The reasoning was not, and the mechanism
    was worse than useless: Match returns REFUTES for ANY grounded string that
    is not the claim's subject, so "Virginia Company", "1565" or "the Atlantic
    Ocean" would all have produced the same confident answer.

    HOW IT CHECKS, AND WHY NOT WITH NER. spaCy would be the obvious tool —
    reject the answer if it is tagged PERSON. It is also the wrong tool here:
    this project's own logs show en_core_web_sm calling *Jamestown* a PERSON,
    so it would very likely call *Saint Augustine* — a city named after a saint
    — a PERSON too, and reject the correct answer.

    So the check is positional instead of categorical. The query carries its own
    type word ('settlement'), and a passage that answers the question will name
    the answer IN THE SAME SENTENCE as that word:

        "...to establish a permanent SETTLEMENT in what became the United
         States, at SAINT AUGUSTINE, Florida (1565)."

    A sentence about an explorer's voyages contains the name but not the type
    word, so it fails. No tagger involved, and nothing to be wrong about.

    FAILS OPEN. If the query has no usable type word, the check is skipped
    rather than rejecting everything — an unverifiable answer should fall
    through to NOT ENOUGH INFO by the normal route, not be blocked here.
    """
    type_word = _type_word(query)
    if not type_word:
        return True

    answer_low = answer.lower()
    for chunk in chunks:
        for sentence in _sentences(chunk):
            low = sentence.lower()
            if type_word in low and answer_low in low:
                return True

    print(f"[Program] Question('{query}') -> '{answer}' REJECTED — never appears "
          f"in a sentence with '{type_word}', so it is probably not a "
          f"{type_word} at all.")
    return False


def _type_word(query: str) -> str | None:
    """
    The noun in the predicate query that names what is being ranked.

    'earliest European permanent settlement United States' -> 'settlement'

    Taken from the parse rather than by position: the last token is 'States'
    here, which names the SCOPE of the superlative, not its TYPE. The head noun
    of the first noun chunk is what the superlative modifies.
    """
    doc = ner.NLP(query)
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in ("NOUN", "PROPN"):
            return chunk.root.text.lower()
    for token in doc:
        if token.pos_ == "NOUN":
            return token.text.lower()
    return None


def _sentences(text: str) -> list[str]:
    """Cheap sentence split. No spaCy: this runs over long evidence chunks."""
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _handle_match(answer: str | None, subject: str) -> tuple[str, str]:
    """
    Match(subject, answer) -> (label, rationale).

    Substring either way, because the two strings name the same thing at
    different lengths: 'Jamestown' against 'the Jamestown settlement', or
    'Saint Augustine' against 'Saint Augustine, Florida'.
    """
    if answer is None:
        return ("NOT ENOUGH INFO",
                "The evidence does not name which entity actually holds this "
                "distinction, so the claim could not be checked against it.")

    a, s = answer.lower(), subject.lower()
    if a in s or s in a:
        return ("SUPPORTS",
                f"The evidence names '{answer}' as holding this distinction, "
                f"which matches the claim's subject '{subject}'.")

    return ("REFUTES",
            f"The evidence names '{answer}' as holding this distinction, not "
            f"'{subject}' as the claim asserts.")


_NUM = r'(\d[\d,]*(?:\.\d+)?)'
_UNIT = r'(' + '|'.join(sorted(_UNIT_TO_BASE, key=len, reverse=True)) + r')'
_QUANTITY_RE = re.compile(_NUM + r'\s*' + _UNIT + r'\b', re.I)


def _handle_quantity(entity: str, dimension: str, family: str):
    """
    Quantity(entity, dimension) -> (value_in_base_units, display, chunks, url)

    THE NUMBER MUST SHARE A SENTENCE WITH THE DIMENSION WORD.

    The first version scored candidates by character distance to the nearest
    dimension keyword anywhere in the chunk. On the Nile that produced:

        Quantity('Nile', length) -> 20 ft        (the Nile is 6,650 km)

    — a measurement of something else entirely, sitting a few hundred characters
    from the word 'length'. Character distance across a 500-character chunk is
    not a relation between a number and a word; it is a coincidence with a
    threshold. A shared sentence is an actual relation:

        "The Nile is about 6,650 km (4,130 mi) long."

    Among candidates that clear that bar, one that appears in a sentence NAMING
    THE ENTITY wins, because an article about the Nile discusses many things'
    lengths and only some sentences are about the Nile's.

    The final tiebreak prefers the LARGEST value, which is a heuristic and worth
    knowing about: in a sentence giving a figure twice in different units
    ("6,650 km (4,130 mi)") both convert to nearly the same base value, so the
    choice is harmless; where a sentence mixes a feature's length with a smaller
    secondary measurement, the larger is usually the headline figure. It is the
    weakest rule here and the first thing to suspect if a comparison looks odd.
    """
    chunks, url = _retrieve(f"{entity} {dimension}", QUESTION_TOP_K)
    if not chunks:
        return None, "", [], ""

    keywords = {dimension} | {k for k, v in _DIMENSIONS.items() if v[0] == dimension}
    entity_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", entity) if len(t) >= 4]

    best = None
    for chunk in chunks:
        for sentence in _sentences(chunk):
            low = sentence.lower()
            if not any(k in low for k in keywords):
                continue
            names_entity = any(t in low for t in entity_tokens) if entity_tokens else False

            for m in _QUANTITY_RE.finditer(sentence):
                unit = m.group(2).lower()
                if _UNIT_FAMILY.get(unit) != family:
                    continue
                value = float(m.group(1).replace(",", "")) * _UNIT_TO_BASE[unit]
                # Lower sorts first: entity-naming sentences beat others, then
                # larger values beat smaller ones.
                rank = (0 if names_entity else 1, -value)
                if best is None or rank < best[0]:
                    best = (rank, value, m.group(0), sentence)

    if best is None:
        print(f"[Program] Quantity('{entity}', {dimension}) — no value found in any "
              f"sentence mentioning {sorted(keywords)}.")
        return None, "", chunks, url

    _, value, display, sentence = best
    print(f"[Program] Quantity('{entity}', {dimension}) -> {display} ({value:g} base)")
    print(f"[Program]   from: {sentence.strip()[:110]}")
    return value, display, chunks, url


def _handle_compare(a, b, subject: str, other: str, dimension: str,
                    disp_a: str, disp_b: str) -> tuple[str, str]:
    """Compare(q_1, q_2) -> (label, rationale). Arithmetic, in Python."""
    if a is None or b is None:
        missing = subject if a is None else other
        return ("NOT ENOUGH INFO",
                f"The {dimension} of '{missing}' could not be read from the "
                f"retrieved evidence, so the comparison could not be made.")

    # IMPLAUSIBLE RATIOS ARE EXTRACTION ERRORS, NOT FINDINGS.
    #
    # Two things compared on the same dimension are, in any claim a person
    # would bother writing, the same order of magnitude — nobody asks whether a
    # river is longer than a shoelace. An observed run read the Nile's length as
    # '20 ft' (6 m) against a genuine 6,650 km: a ratio of a million, which is
    # not a fact about rivers but a fact about the extractor.
    #
    # 1000x is deliberately loose. It passes every real comparison (Everest is
    # 1.03x K2; the Burj is 1.9x the Empire State) and catches the failures,
    # which are wrong by several orders of magnitude rather than by a little.
    ratio = max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")
    if ratio > 1000:
        return ("NOT ENOUGH INFO",
                f"The values read from the evidence — {subject} {disp_a}, "
                f"{other} {disp_b} — differ by a factor of {ratio:.0f}, which "
                f"means at least one was misread. No comparison was made.")

    if a > b:
        return ("SUPPORTS",
                f"The evidence gives {subject} as {disp_a} and {other} as "
                f"{disp_b}, so the claim holds.")
    if a < b:
        return ("REFUTES",
                f"The evidence gives {subject} as {disp_a} and {other} as "
                f"{disp_b}, so the claim is false.")
    return ("NOT ENOUGH INFO",
            f"The evidence gives both {subject} and {other} as {disp_a}, which "
            f"does not settle the comparison.")


# ── Execution ─────────────────────────────────────────────────────────────────
def execute(program: Program, claim: str) -> dict | None:
    """
    Run a program and return a result dict shaped like grounded_verify's.

    Returns None if execution could not proceed at all, which sends the caller
    back to single-shot verification rather than leaving the claim unchecked.
    """
    if program.kind == "superlative":
        category = program.steps[0].args[0]
        subject = program.steps[1].args[1]

        answer, chunks, url = _handle_question(category)
        label, rationale = _handle_match(answer, subject)

        program.trace = [
            f"answer_1 = Question('{category}') -> {answer or 'UNKNOWN'}",
            f"label = Match('{subject}', answer_1) -> {label}",
        ]
        program.chunks, program.source_url = chunks, url

    elif program.kind == "comparative":
        subject, dimension, family = program.steps[0].args
        other = program.steps[1].args[0]

        a, disp_a, chunks_a, url_a = _handle_quantity(subject, dimension, family)
        b, disp_b, chunks_b, _ = _handle_quantity(other, dimension, family)

        label, rationale = _handle_compare(
            a, b, subject, other, dimension, disp_a, disp_b
        )

        program.trace = [
            f"q_1 = Quantity('{subject}', {dimension}) -> {disp_a or 'UNKNOWN'}",
            f"q_2 = Quantity('{other}', {dimension}) -> {disp_b or 'UNKNOWN'}",
            f"label = Compare(q_1, q_2) -> {label}",
        ]
        program.chunks = chunks_a + chunks_b
        program.source_url = url_a

    else:
        return None

    for line in program.trace:
        print(f"[Program]   {line}")

    return {
        "claim": claim,
        "label": label,
        # The trace travels with the verdict. A program-guided answer that
        # cannot show its steps has thrown away the reason for using one.
        "rationale": rationale + "  [program: " + " ; ".join(program.trace) + "]",
        "evidence": program.chunks[0] if program.chunks else "",
        "nei_kind": None,
        "ner_query": program.query,
        "source_url": program.source_url,
        "program_kind": program.kind,
        "timings": {"retrieval_s": 0.0, "verification_s": 0.0},
    }


def try_verify(claim: str) -> dict | None:
    """
    Entry point. Returns a result dict, or None when no program applies and the
    caller should fall back to single-shot verification.
    """
    if not PROGRAM_MODE:
        return None

    program = synthesize(claim)
    if program is None:
        return None

    print(f"\n[Program] {program.kind.upper()} claim — running program:")
    for step in program.steps:
        print(f"[Program]   {step.var} = {step.op}{step.args}")

    try:
        return execute(program, claim)

    except (AttributeError, TypeError, NameError, KeyError, IndexError) as e:
        # A BUG, NOT A RUNTIME CONDITION — and the two must not look alike.
        #
        # The first release of this module called a helper that exists in some
        # versions of ollama_client and not others. Every program run raised
        # AttributeError, the blanket handler caught it, and the log line was
        # indistinguishable from "no program applied". The feature was dead for
        # a full 700-second run and the only trace was one quiet line.
        #
        # Falling back is still right — losing the claim would be worse — but
        # these exception types mean the code is broken, so say so loudly.
        print(f"[Program] *** BUG: {type(e).__name__}: {e}")
        print(f"[Program] *** This is a defect in program.py, not a retrieval "
              f"failure. The claim falls back to the verifier, but the program "
              f"path is NOT running.")
        return None

    except Exception as e:
        # Expected runtime trouble: a network timeout, a Wikipedia error page,
        # a malformed response. Fall through quietly; the verifier can still
        # answer from its own retrieval.
        print(f"[Program] Execution failed ({type(e).__name__}: {e}) — "
              f"falling back to the verifier.")
        return None
