import re
import time

import requests

import models.ollama_client as ollama_client
import Pipeline.ner as ner
import Pipeline.retriever as retriever
import Pipeline.verifier as verifier

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
FAST_MODEL   = "phi3:mini"   # swap with your chosen fast model name in Ollama

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 10   # safety ceiling so the loop never runs forever

# Keep phi3 resident between calls rather than reloading it after Ollama's
# 5-minute default. Measured cost of a cold load on this hardware: 7.7s.
KEEP_ALIVE = "30m"

# How many consecutive rejected/malformed responses to tolerate before giving
# up on the current fact slot. Under greedy decoding the model is deterministic
# given a prompt, so an unchanged prompt cannot produce a new answer; we vary
# the prompt by feeding the rejection back, but if it still will not comply
# after being shown its own failures this many times, it will not comply at all.
MAX_CONSECUTIVE_REJECTIONS = 3

# Per-sentence iteration ceiling. A single sentence holds far fewer atomic
# facts than a document, so this can be much tighter than MAX_ITERATIONS —
# which now bounds total work across the whole input instead.
MAX_ITERATIONS_PER_SENTENCE = 5

# CLAUSE_MODE splits each sentence into clause-level propositions with spaCy
# and renders one claim per clause, instead of asking the model to search for
# "the next uncovered relation" until it says Terminate.
#
# Set False to restore the search loop — that is the comparison baseline, and
# the two differ in both coverage and cost, so it is worth measuring rather
# than assuming.
#
# NOW ALL-OR-NOTHING. This used to apply only to sentences that enumerate to
# two or more clauses; single-clause sentences quietly used the search loop
# regardless of the flag. That hybrid is what lost a claim: the search-loop
# prompt shows the document as context beside a target sentence and asks the
# model to pick what to extract, and given a target opening with an unresolved
# pronoun it picked from the context instead. A single clause needs no
# enumeration, but it still benefits from being ASKED THE NARROWER QUESTION —
# "rewrite this clause" rather than "find a fact somewhere in here".
#
# So the flag now selects the path for every sentence, which also makes the two
# configurations cleanly comparable: with it False nothing reaches
# _extract_from_clauses, with it True nothing reaches _extract_from_sentence.
CLAUSE_MODE = True

# Whether the extraction history carries the verifier's Answer/Rationale lines
# (AFEV Eq. 2) or only the bare list of covered facts.
#
# Set False by default on the evidence of an observed run: after a REFUTES
# verdict, the extractor's very next output was the verifier's own correction
# restated as a fact ("The Bosporus forms the continental boundary between Asia
# and Europe"), which the faithfulness gate then had to reject. Eq. 2 was meant
# to aid coreference, but rule 4 now resolves references from the source
# document instead, leaving the rationales with no extraction purpose and a
# demonstrated cost. Flip to True to reproduce the paper's configuration and
# measure the difference rather than taking this on faith.
INCLUDE_RATIONALES_IN_HISTORY = False

# Function words the faithfulness gate tolerates in an extracted fact even when
# they do not occur in the source text. Only closed-class items belong here:
# auxiliaries, determiners, relativisers, connectives. Anything carrying
# referential content (a name, place, number, date) must come from the input.
# Words of 3 characters or fewer are filtered out before the check, so short
# items such as 'is', 'was', 'the', 'and', 'of' need no entry.
# NOTE UNDER "lemma_keep_propn": four entries here — were, been, being, does —
# are unreachable, because they lemmatise to 'be' and 'do', which fall below
# MIN_CONTENT_WORD_LENGTH and are dropped before this set is ever consulted.
# Harmless (they are filtered either way) but worth knowing before adding more:
# entries should be LEMMAS in this mode, not inflected forms.
FUNCTION_WORDS = {
    "were", "been", "being", "have", "having", "does", "will", "would",
    "that", "this", "these", "those", "which", "whose", "there", "then",
    "they", "them", "their", "with", "within", "from", "into", "also",
    "than", "when", "while", "where", "both", "each", "such", "same",
    "some", "more", "most", "other", "another", "after", "before",
    "between", "during", "about", "only", "very", "much", "many",
}


_ARTICLES = {"the", "a", "an"}


def _normalize_claim(fact: str) -> str:
    """
    Canonical form used ONLY for duplicate detection — never for display,
    retrieval, or verification, all of which need the claim verbatim.

    Beyond lowercasing and collapsing whitespace, this drops articles and
    punctuation: differences that carry no propositional content. An observed
    run verified "...with Yavuz Sultan Selim Bridge." and "...with the Yavuz
    Sultan Selim Bridge." as two separate claims, at 118s and 130s, because
    the previous normalizer compared raw lowercased strings.

    Deliberately exact-match rather than similarity-based. A threshold would
    also catch paraphrases ("Bosporus is a strait located between X and Y"),
    but claims that differ in one content word can be genuine opposites —
    "between Africa and Europe" vs "between Asia and Europe" overlap almost
    entirely — and silently merging those would defeat the whole pipeline.
    """
    words = re.findall(r"[a-z0-9çğışöü]+", fact.lower())
    return " ".join(w for w in words if w not in _ARTICLES)


# ── Tunable parameters of the faithfulness vocabulary ─────────────────────────
# These were magic numbers inside _content_words. Naming them matters for more
# than tidiness: a threshold buried in a comprehension is a decision nobody
# made, nobody documents, and nobody can sweep. You cannot tune what you cannot
# name.
#
# MIN_CONTENT_WORD_LENGTH excludes short function words so that 'the', 'and',
# 'of' never count as material the claim borrowed. It also has a consequence
# that was invisible while it was an inline `len(w) <= 3`:
#
#   'not' IS THREE CHARACTERS, so the faithfulness gate cannot see negation.
#
# An observed run turned "It is universally acknowledged as longer than the
# Nile" into "The Nile is NOT universally acknowledged as the world's longest
# river" — a complete reversal of meaning — and the gate passed it, because
# every surviving word did appear in the source and the inserted 'not' was
# filtered out before comparison. Lowering this threshold is NOT the fix
# (it would flood the set with articles and prepositions); negation needs its
# own dedicated check. The constant is named here so the limitation is visible
# at the point where it is created.
MIN_CONTENT_WORD_LENGTH = 4

# Suffixes stripped so that inflectional variants introduced by rewriting a
# clause ('separates' vs 'separated') do not register as foreign material.
# Order matters — longest first, first match wins.
_STEM_SUFFIXES = ("ing", "es", "ed", "s")

# A stem shorter than this is too mangled to compare, so the suffix is kept.
# Without it, 'axes' would stem to 'ax' and collide with unrelated words.
MIN_STEM_LENGTH = 4

# ── Stemming backend ──────────────────────────────────────────────────────────
# STEMMER_LANGUAGE is the first piece of groundwork for non-English input. It is
# groundwork only: sentence segmentation, NER and dependency parsing all come
# from en_core_web_sm, the retriever is pinned to en.wikipedia.org, and both
# models are English-tuned. Changing this constant alone will not make the
# pipeline work in Turkish — it removes one of several blockers.
#
# Snowball supports ~28 languages including Turkish, though Turkish is
# agglutinative ("ev → evler → evlerim → evlerimde") and any suffix-stripping
# algorithm handles it poorly; a real Turkish deployment wants a morphological
# analyser such as Zeyrek or Zemberek, not a stemmer.
# ── How content words are normalised ──────────────────────────────────────────
# "lemma_keep_propn"  spaCy lemma for ordinary words, proper nouns untouched.
#                     Won the benchmark on every column; see _content_words_lemma.
# "stem"              the previous behaviour, driven by STEMMER_LANGUAGE and
#                     USE_SNOWBALL_STEMMER below. Kept as the comparison
#                     baseline — flip this back and re-run to reproduce the
#                     numbers rather than trusting them.
#
# The stemming constants below still apply in "stem" mode only.
CONTENT_WORD_MODE = "lemma_keep_propn"

STEMMER_LANGUAGE = "english"

# Snowball is MORE AGGRESSIVE than the four-suffix stripper it replaces:
#   separates → separ      continental → continent      forms → form
#
# That cuts both ways for the faithfulness gate. More inflectional variants
# collapse together, so fewer faithful rewrites are wrongly rejected — but
# genuinely foreign words also collapse into source words more often, so fewer
# unfaithful claims are caught. An observed run flagged
# ['asia', 'boundary', 'continental', 'form']; under Snowball, 'continental'
# and 'form' would disappear from that set if the document mentions a
# "continent" or "forms" anywhere.
#
# The gate's job is precision, so this is a trade to MEASURE, not to assume.
# Flip to False to restore the previous behaviour and compare.
USE_SNOWBALL_STEMMER = True


def _suffix_stem(word: str) -> str:
    """Original four-suffix stripper. Retained as the no-NLTK fallback."""
    for suffix in _STEM_SUFFIXES:
        if len(word) - len(suffix) >= MIN_STEM_LENGTH and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _build_stemmer():
    """
    Resolve the stemming function once at import.

    Falls back to _suffix_stem when NLTK is absent or the language is
    unsupported, so the pipeline keeps working on a machine that has not
    installed it — the gate degrades in strictness rather than crashing.
    """
    if not USE_SNOWBALL_STEMMER:
        return _suffix_stem

    try:
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer(STEMMER_LANGUAGE)
        print(f"[Extractor] Stemmer: Snowball ({STEMMER_LANGUAGE}).")
        return stemmer.stem
    except Exception as e:
        print(f"[Extractor] Snowball unavailable ({e}); using suffix stripper.")
        return _suffix_stem


_stem = _build_stemmer()


def _content_words(text: str) -> set[str]:
    """
    Lowercase content words of `text`, crudely stemmed.

    Stemming is deliberately naive — it exists so that inflectional variants
    introduced by rewriting a clause into a standalone sentence ('separates'
    vs 'separated', 'prizes' vs 'prize') do not register as foreign material.
    It is not meant to be linguistically correct; it only needs to leave
    proper nouns and numbers untouched, which it does.

    See MIN_CONTENT_WORD_LENGTH above for why this function is blind to
    negation — a limitation, not an oversight, and one that needs a separate
    check rather than a different threshold.
    """
    if CONTENT_WORD_MODE == "lemma_keep_propn":
        return _content_words_lemma(text)

    words = re.findall(r"[a-zçğışöü0-9]+", text.lower())
    return {
        _stem(w) for w in words
        if len(w) >= MIN_CONTENT_WORD_LENGTH
    }


def _content_words_lemma(text: str) -> set[str]:
    """
    Lemmatise ordinary words; leave proper nouns exactly as written.

    Chosen on benchmark evidence (benchmarks/stemmer_bench.py, 167 comparison
    points across six candidates):

        candidate                  A collapse  B distinct   C propn   FA  FR
        suffix_stem                      26%        100%       14%     0   4
        snowball                         78%         76%       30%     1   3
        porter                           76%         74%       10%     1   3
        lancaster (control)              80%         44%        6%     1   3
        spacy_lemma                      96%        100%       50%     0   2
        spacy_lemma + propn skip         96%        100%       92%     0   2

    The C column is why. Porter and Snowball are string rewriting with no
    concept of a name, so step 1a strips a terminal 's' and every place name
    ending in -s becomes a plural: Paris->pari, Athens->athen, Wales->wale.
    Snowball only spares Bosporus because Porter2 exempts words ending in 'us'.
    Fact-checking text is dense with such names, and a mangled name silently
    stops matching its own mention in the evidence.

    No pure stemmer can do better — once the string is lowercased, 'Paris' and
    'parries' are indistinguishable. The part-of-speech tag is the missing
    information, and spaCy already computes it for this text elsewhere.

    Cost is a parse (~1ms for a claim, ~20ms for a document) instead of a
    regex. Against ~19s extraction calls that is not measurable, and
    extract_atomic_facts parses the document once per request, not per claim.
    """
    doc = ner.NLP(text)

    out: set[str] = set()
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        surface = token.text.lower()
        if len(surface) < MIN_CONTENT_WORD_LENGTH:
            continue

        # PROPN passes through untouched. This is the whole point: a name is
        # not an inflected form of anything, so any normalisation of it is
        # damage.
        out.add(surface if token.pos_ == "PROPN" else token.lemma_.lower())

    return out

# ── Text you are testing with ─────────────────────────────────────────────────
text_ba = "Here is a 500-word text..."   # keep your original text here

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Precondition (your original function, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def precondition(text):
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split(' ')
    if len(words) > 500:
        print("Validation Failed: Text exceeds 500 words.")
        return None

    allowed_pattern = r'^[a-zA-ZçğışöüÇĞİŞÖÜ0-9\s.,!?\-\'\"()]+$'
    if not re.match(allowed_pattern, text):
        print("Validation Failed: Text contains invalid characters.")
        return None

    print(f"Preprocessed text: {text}")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Call Ollama (shared helper)
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """Send a prompt to the fast model and return the raw response string."""
    payload = {
        "model": FAST_MODEL,
        "prompt": prompt,
        "stream": False,
        # Keep the model resident; see the note on KEEP_ALIVE in verifier.py.
        "keep_alive": KEEP_ALIVE,
        "options": {
            # Explicit context window to prevent 500s on long iterative prompts.
            "num_ctx": 4096,
            # Greedy decoding. Ollama defaults to temperature 0.8 / top_p 0.9,
            # which is actively harmful here: extraction is a copying task, so
            # the faithful span is the argmax and every unit of sampling noise
            # is an opportunity for the model to drift toward the fluent,
            # world-consistent paraphrase it prefers over the literal claim.
            # Fixing the seed also makes runs reproducible, without which no
            # prompt change can be evaluated.
            #
            # This read 0.2 while the comment above claimed 0. Behaviour was
            # unaffected — top_k=1 restricts the candidate set to the argmax, and
            # temperature cannot reorder a set of one — so the file simply
            # documented a setting it did not use. That is a trap rather than a
            # bug: relaxing top_k later for output diversity would silently make
            # 0.2 live sampling, and reproducibility would die with no edit
            # anywhere near the change. Set to 0 so the two agree and the
            # greedy guarantee survives independently of top_k.
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": 0,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    body = response.json()
    ollama_client.log_inference_stats(body, FAST_MODEL)
    return body["response"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Parse numbered list from model output (STRICT format contract)
# ─────────────────────────────────────────────────────────────────────────────
# Sentinel distinguishing "model signaled Terminate" (stop the loop) from
# "response was malformed" (reject this iteration, but keep looping).
# The old parser conflated both into None, and its loose regex matched the
# word 'Fact' anywhere in the response — which let leaked chain-of-thought
# (e.g. '...we proceed with Fact-2 extraction') be accepted as a claim.
TERMINATE = "__TERMINATE__"

def parse_fact_from_response(response: str, expected_idx: int) -> str | None:
    """
    Strictly parse one atomic fact from model output.

    Accepted formats (nothing else):
        - A line that is exactly 'Terminate' (optional trailing punctuation)
          → returns TERMINATE sentinel.
        - A line starting with 'Fact_<expected_idx>:' (underscore/dash/space
          separators tolerated) → returns the fact text after the colon.

    Anything else → None (malformed; caller rejects the iteration).
    Requiring the *expected* index prevents the model from re-emitting an
    earlier fact or inventing out-of-sequence ones.
    """
    # PRECEDENCE: a valid Fact_N line wins over Terminate. phi3 sometimes
    # emits both in one response ("Fact_2: ... \n ... \n Terminate");
    # checking Terminate first threw the fact away and closed the loop
    # with the input only partially covered.
    match = re.search(
        rf'(?im)^\s*Fact[_\s\-]?{expected_idx}\s*[:\-]\s*(.+?)\s*$',
        response,
    )
    if match:
        fact = match.group(1).strip()

        # Strip trailing parenthetical meta-commentary phi3 likes to append,
        # e.g. "... in 1889. (The entity ... extracted from the text.)" —
        # it pollutes the NER query and the verifier prompt.
        fact = re.sub(r'\s*\([^()]*\)[\s.]*$', '', fact).strip()

        if fact:
            return fact

    # Terminate must be a standalone line — not merely the substring
    # 'terminate' anywhere, which false-positived on claims containing it.
    if re.search(r'(?im)^\s*terminate[.!]?\s*$', response):
        print("[Extractor] Termination signaled. Loop closed successfully.")
        return TERMINATE

    return None  # malformed output — no fallback acceptance


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build extraction prompt  (mirrors Figure 3 in the AFEV paper)
# ─────────────────────────────────────────────────────────────────────────────
def build_extraction_prompt(
    original_text: str,
    extracted_facts: list[str],
    verification_results: list[str],
    verification_rationales: list[str],  # Crucial addition from Eq. 2
    rejected_attempts: list[tuple[str, str, str]] | None = None,
    target_sentence: str | None = None,
    target_clause: str | None = None,
) -> str:
    """
    Dynamic iterative extraction prompt that mirrors Section 4.1 & 5.7 of AFEV.

    `rejected_attempts` carries (attempt_text, foreign_words) pairs for outputs
    the faithfulness gate refused this round. It exists because the extractor
    now runs at temperature 0: greedy decoding is a deterministic function of
    the prompt, so re-sending an unchanged prompt after a rejection reproduces
    the rejected output *exactly*, forever. One observed run burned iterations
    2 through 10 re-emitting the same sentence verbatim. Feeding the rejection
    back is what makes the retry a different question rather than a replay —
    and it turns the gate from a silent filter into a correction signal.

    `target_sentence` narrows what may be extracted while leaving the full
    document visible. The two roles are deliberately distinct:

        original_text   — CONTEXT. Read-only. Exists so pronouns resolve.
        target_sentence — TARGET.  The only source of new facts this round.

    Both are needed. Scoping to a sentence bounds the decomposition problem and
    stops one uncooperative sentence from consuming the whole budget, but a
    sentence read in isolation is often unverifiable: "You can walk between
    them with Yavuz Sultan Selim Bridge" has no recoverable meaning without the
    sentence before it. Keeping the document in view is what lets "them"
    resolve while the extraction stays scoped.
    """
    # Each rejection carries the KIND of defect, because the advice that fixes
    # one makes the others worse. An earlier version rendered every rejection
    # with the faithfulness wording, so a claim rejected for an unresolved
    # pronoun was told "these words do not appear in the text: 'its' refers to
    # something outside the claim" — an assertion that is both false ('its' is
    # in the text) and self-contradictory. phi3 responded by re-emitting the
    # identical sentence three times, which under greedy decoding is exactly
    # what incoherent feedback should be expected to produce.
    _ADVICE = {
        "faithfulness": "Rewrite using ONLY wording that appears in the document.",
        "decontextualization": "Replace the reference with the name it stands for, "
                               "taken from the document.",
        "fluency": "Write one complete, grammatical declarative sentence.",
        "malformed": "Respond with exactly one line: 'Fact_N: <sentence>' or 'Terminate'.",
        # UNREACHABLE as of the first-repeat stop. Both extraction paths now
        # `break` on a duplicate instead of feeding it back, so nothing appends a
        # ("duplicate", ...) rejection any more. Kept rather than deleted because
        # the branch that produced it is one flag away from returning — if the
        # per-sentence/per-document distinction in option B is ever built, a
        # duplicate of an EARLIER SENTENCE's fact becomes a retryable case again
        # and this wording is what it needs. Delete both this entry and the
        # matching branch below if that never happens.
        "duplicate": "Extract a DIFFERENT relation from the sentence — a different "
                     "verb, date, place or property that no earlier fact states. "
                     "If the sentence truly contains only one relation, output "
                     "Terminate.",
    }

    rejection_block = ""
    if rejected_attempts:
        lines = []
        for attempt, kind, detail in rejected_attempts:
            if kind == "faithfulness":
                reason = f"These words are not in the document: {detail}."
            elif kind == "malformed":
                reason = "This was not in the required output format."
            elif kind == "duplicate":
                reason = "REPEAT. You already extracted this exact fact."
            else:
                reason = detail if detail.endswith(".") else f"{detail}."
            lines.append(f'- You wrote: "{attempt}"\n  REJECTED. {reason}')

        # Only advise on defects actually seen, so the model is not handed four
        # competing instructions when it made one mistake.
        advice = [_ADVICE[k] for k in dict.fromkeys(k for _, k, _ in rejected_attempts)]

        rejection_block = (
            "\nFAILED ATTEMPTS — do not repeat these:\n"
            + "\n".join(lines)
            + "\n" + " ".join(advice)
            + " If nothing extractable remains, output Terminate.\n"
        )

    # Presentation of the source material. When scoped to a sentence the
    # document is shown first and explicitly demoted to context, so the model
    # can resolve references without treating the whole document as fair game.
    scoped = bool(target_sentence) and target_sentence.strip() != original_text.strip()
    if scoped:
        text_block = (
            'Full document — CONTEXT ONLY. Use it to resolve pronouns and vague\n'
            'references. Do NOT extract facts from it.\n'
            f'\"\"\"{original_text}\"\"\"\n\n'
            'TARGET SENTENCE — extract only from this sentence:\n'
            f'\"\"\"{target_sentence}\"\"\"'
        )
        scope_rule = (
            "- SCOPE: Extract only what the TARGET SENTENCE asserts. If it contains a pronoun "
            "or vague reference, replace it with the matching wording from the full document.\n"
        )
    else:
        text_block = f'Text:\n\"\"\"{original_text}\"\"\"'
        scope_rule = ""

    # ── CONSTANT HEADER ───────────────────────────────────────────────────────
    # Byte-identical on EVERY call, first iteration or fiftieth, scoped or not.
    #
    # The rules are NUMBERED and lead with atomicity because compressing them
    # into terse prose caused a measured regression: the extractor produced a
    # claim that fused two relations, and once the extra clause was in the
    # claim its tokens dominated retrieval, pulling in evidence that confirmed
    # the true half while the article refuting the other half never surfaced.
    # A correct REFUTES became SUPPORTS.
    #
    # Atomicity is upstream of retrieval and of the verdict. A compound claim
    # whose halves have different truth values cannot receive a correct label,
    # whatever the verifier does.
    #
    # THE WORKED EXAMPLES BELOW MUST NOT COME FROM THE TEST SET. An earlier
    # version used a Jamestown sentence that is also a regression case, so the
    # model was being shown the answer to a question it was about to be asked —
    # extraction on that input then measured nothing. Keep the examples in
    # domains the evaluation inputs never touch.
    #
    # This used to be two different headers — "You are an expert Atomic Fact
    # Extractor" for iteration 1 and "You are an adaptive, iterative Atomic
    # Fact Extractor" for the rest — which meant the two shared no prefix at
    # all and every transition from iteration 1 to 2 recomputed the whole
    # prompt. Ollama's measured cost of that miss versus a hit, same model and
    # same prompt size, in one run:
    #
    #     prompt 575 tok in 22.6s  ( 25 tok/s)   <- cache miss
    #     prompt 595 tok in  1.4s  (423 tok/s)   <- cache hit
    #
    # 17x, for reordering text. Anything that varies — the document, the target
    # sentence, the history, the rejections — now goes below.
    header = f"""You are an expert Atomic Fact Extractor. You extract ONE atomic fact per response.

Rules, in priority order:

1. ATOMICITY — ONE RELATION ONLY. A fact states a single subject-relation-object
   link and nothing more. If a sentence joins two statements with "and", ",",
   "which", "but" or "because", those are SEPARATE facts — extract one now and
   leave the other for the next round. NEVER join two relations with "and" into
   one fact.

   Sentence: "The museum opened in 1932 and holds over 400 paintings."
   CORRECT   Fact_1: The museum opened in 1932.
   WRONG     Fact_1: The museum opened in 1932 and holds over 400 paintings.
   The wrong version fuses two relations, so one verdict cannot describe both.

2. NEVER EXTEND A PREVIOUS FACT. Each new fact must state a relation none of the
   earlier facts already state. Adding a clause to a fact you already produced is
   not a new fact — it is the same fact, and it is wrong.

3. FAITHFULNESS — Use ONLY information stated in the document. Copy names, dates
   and places exactly as written. NEVER add dates, names, numbers or details from
   your own memory, even if you are certain they are true. If the document says
   "in 1889", write "in 1889" — never a more specific date.

4. DECONTEXTUALIZATION — the fact must stand alone with no access to the
   document. Replace every backward-pointing reference (it, they, them, he, she,
   this, that city, there, then) with the name it stands for, exactly as that
   name appears in the document. Make time and place absolute where the document
   states them. Generic "you" or "one" refers to nobody in particular — leave it.
   If a reference has no antecedent in the document, leave it unchanged rather
   than inventing one.

5. TERMINATION — if every relation has already been extracted, output exactly:
   Terminate

6. OUTPUT FORMAT — STRICT: exactly ONE line, either
   "Fact_N: <one plain declarative sentence>" or "Terminate". NEVER both.
   No Answer lines, no Rationale lines, no arrow notation (->), no parentheses,
   no explanations of any kind.

Example of a CORRECT extraction:
Document: \"\"\"Marie Curie, who was born in Warsaw, won two Nobel Prizes.\"\"\"
Fact_1: Marie Curie was born in Warsaw.

Example of an INCORRECT extraction — never do this:
Document: \"\"\"Marie Curie, who was born in Warsaw, won two Nobel Prizes.\"\"\"
Fact_1: Marie Curie was born on 7 November 1867 in Warsaw, Poland.
This is wrong because the exact date and the country are NOT in the document — they were added from memory.
"""
    # NOTE: the cache boundary is HERE, at the end of `header`. It is marked in
    # this comment and NOT in the prompt text.
    #
    # A previous version printed "--- Everything above this line is identical on
    # every call..." into the prompt itself. phi3 treated that meta-text as
    # content to continue and echoed the whole variable section back — 276
    # output tokens instead of 22, 31.5s instead of 2s. Explanatory scaffolding
    # about a prompt does not belong inside the prompt; the model cannot tell
    # the difference between instructions and commentary.

    # ── CLAUSE MODE ───────────────────────────────────────────────────────────
    # One clause in, one claim out. The model is no longer asked to FIND a fact
    # — syntax already did that — only to render a clause the enumerator handed
    # it as a standalone sentence.
    #
    # This removes the loop's termination problem. Previously the only way to
    # learn a sentence was exhausted was for phi3 to output "Terminate", which
    # it does not do; instead it repeated its last fact, costing three wasted
    # calls per sentence before the rejection cap fired. With a clause list the
    # loop is bounded before it starts.
    if target_clause:
        idx = len(extracted_facts) + 1
        return f"""{header}
{text_block}

CLAUSE TO REWRITE:
\"\"\"{target_clause}\"\"\"

Rewrite the CLAUSE above as one standalone declarative sentence. Do not add
information, do not drop information, and do not extract anything from the rest
of the document — the clause is already the single relation to state. Resolve
any pronoun in it using the document. If the clause is not a checkable factual
statement, output exactly: Terminate
{rejection_block}
Fact_{idx}:"""

    # 1. Base Strategy: First Iteration
    if not extracted_facts:
        return f"""{header}
{scope_rule}
{text_block}
{rejection_block}
Extract Fact_1:"""

    # 2. Dynamic Strategy: Subsequent Iterations (The Feedback Loop)
    #
    # Built with a list comprehension + str.join instead of repeated `+=`.
    # Every `+=` on a str allocates a whole new string (str is immutable), so
    # the old loop was quadratic in the history length; join walks the parts
    # once and allocates the result buffer exactly once.
    #
    # strict=True makes a length mismatch between the three parallel lists a
    # loud ValueError instead of a silent truncation: plain zip() stops at the
    # shortest input, so one missing rationale would quietly drop a verified
    # fact out of the history the extractor reasons over.
    #
    # INCLUDE_RATIONALES_IN_HISTORY gates the Answer/Rationale lines. With them
    # on, an observed run answered a REFUTES rationale by proposing the
    # verifier's correction as the next "fact" — the rationale is the single
    # strongest pull away from faithful extraction in the system, because it
    # sits near the end of the prompt in fluent prose that contradicts the
    # source. With them off, the extractor still sees which parts of the text
    # are covered, which is all it needs to decide what remains.
    if INCLUDE_RATIONALES_IN_HISTORY:
        history_block = "\n\n".join(
            f"Fact_{i}: {fact}\nAnswer_{i}: {label}\nRationale_{i}: {rationale}"
            for i, (fact, label, rationale) in enumerate(
                zip(
                    extracted_facts,
                    verification_results,
                    verification_rationales,
                    strict=True,
                ),
                1,
            )
        )
    else:
        history_block = "\n".join(
            f"Fact_{i}: {fact}" for i, fact in enumerate(extracted_facts, 1)
        )

    next_idx = len(extracted_facts) + 1
    unit = "TARGET SENTENCE" if scoped else "Original Text"

    # Same constant header, then only what varies. The facts already extracted,
    # the document and the rejections all change between iterations, so they sit
    # below the cache boundary.
    prompt = f"""{header}
{scope_rule}
{text_block}

Already extracted (do not repeat these):
{history_block}

Decide: does the list above already cover everything checkable in the {unit}?
If yes, output exactly: Terminate
If no, extract the next atomic fact FROM THE {unit}, as: Fact_{next_idx}: <sentence>
{rejection_block}
Extract Fact_{next_idx} or Terminate:"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Iterative extractor  (core logic from AFEV Section 4.1)
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_nei(fact: str, chunks: list[str]) -> str:
    """
    Distinguish the two very different situations that both surface as NEI.

    'NOT ENOUGH INFO' currently collapses two states that call for opposite
    responses:

      RETRIEVAL_FAILURE — the evidence never mentions what the claim is about,
        so no verdict was ever possible. Recoverable: re-query and try again.
        This is what happened to "Bosporus is located between Africa and
        Europe": the query was built from Africa and Europe alone, the
        Bosporus article was never fetched, and the verifier was asked to
        adjudicate a strait using chunks about African rainfall. Its NEI was
        the correct response to bad evidence, not a statement about the claim.

      GENUINE — the right articles were retrieved and are authentically
        silent or ambiguous on this particular point. Terminal: no better
        query will help.

    The test is deliberately mechanical rather than another LLM call: does the
    claim's subject appear anywhere in the retrieved text? Zero mentions
    across every chunk means the evidence is not about the claim at all.

    Returns "RETRIEVAL_FAILURE", "GENUINE", or "UNKNOWN" when the subject
    could not be recovered (pronoun subject, parse failure) and the question
    cannot be decided this way.
    """
    subject = ner.extract_subject(fact)
    if not subject:
        return "UNKNOWN"

    # Match on the subject's PROPER NOUN, falling back to its longest token.
    #
    # Two previous versions, both wrong, and the second is instructive:
    #
    #   subject.split()[-1]        The head word. Too permissive when the head
    #                              is a common noun — "Yavuz Sultan Selim
    #                              Bridge" matched on 'bridge' against a passage
    #                              comparing tower heights, and "water
    #                              molecules" matched on 'molecules' in an
    #                              article about the properties of water. Both
    #                              reported GENUINE; both were retrieval
    #                              failures.
    #
    #   max(split(), key=len)      The longest token, as a crude proxy for the
    #                              rarest. It fixed those two cases and then
    #                              reproduced the identical bug the moment a
    #                              common noun happened to be longer than the
    #                              name:
    #
    #                                'English settlement of Jamestown'
    #                                 English=7  settlement=10  Jamestown=9
    #                                 -> picked 'settlement'
    #
    #                              which matches essentially any colonial-era
    #                              passage, so the diagnosis was GENUINE by
    #                              construction and told the reader nothing.
    #
    # Length was never the property being reached for. The property is "does
    # this string name ONE thing?", and part of speech answers it directly:
    # a proper noun is a name, a common noun is a category. spaCy has already
    # tagged this text, so the answer is free. Length survives only as the
    # fallback for subjects with no proper noun at all ("water molecules"),
    # where the old proxy is still the best available.
    subject_doc = ner.NLP(subject)
    proper_nouns = [t.text for t in subject_doc if t.pos_ == "PROPN"]

    if proper_nouns:
        head = max(proper_nouns, key=len).lower()
        basis = "proper noun"
    else:
        head = max(subject.split(), key=len).lower()
        basis = "longest token (no proper noun in subject)"

    haystack = " ".join(chunks).lower()

    # BOTH outcomes are logged. Only RETRIEVAL_FAILURE used to print, so a
    # GENUINE diagnosis reached the reader through run()'s summary suffix
    # instead — a different function than the one that made the decision. That
    # made the diagnostic look disconnected from the pipeline when it was
    # merely silent, and it hid which token the decision rested on, which is
    # exactly what was wrong above.
    if head in haystack:
        print(
            f"[Diagnosis] Subject '{subject}' matched on '{head}' ({basis}) in "
            f"{len(chunks)} evidence chunk(s) — the evidence IS about this claim, "
            f"so NEI is a considered verdict."
        )
        return "GENUINE"

    print(
        f"[Diagnosis] Subject '{subject}' — '{head}' ({basis}) appears in 0 of "
        f"{len(chunks)} evidence chunk(s). The evidence is not about this claim. "
        f"NEI is a retrieval failure, not a verdict."
    )
    return "RETRIEVAL_FAILURE"


def grounded_verify(fact: str) -> dict:
    """
    Evidence-grounded verification used inside the extraction feedback loop.

    Replaces lightweight_verify(), which judged facts with phi3's general
    knowledge. That was the pipeline's main hallucination channel: world-
    knowledge rationales (invented company names, people, dates) were fed
    back into the extraction history, and rule 4 of the prompt explicitly
    told the extractor to recycle entities from those rationales into new
    "facts" that never appeared in the input text.

    This calls the same Wikipedia-grounded retriever + verifier used for
    the final answer (AFEV's intended design), so rationales can only
    contain entities present in retrieved evidence. Each fact is now also
    verified exactly once — main.py no longer re-runs verification.

    Returns a complete per-fact result dict ready for postprocessor.process().
    """
    t0 = time.perf_counter()
    evidence_pack = retriever.fetch(fact)
    retrieval_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    verdict = verifier.verify(fact, evidence_pack["chunks"])
    verification_s = time.perf_counter() - t1

    print(
        f"[Timing] retrieval: {retrieval_s:.2f}s | "
        f"verification: {verification_s:.2f}s — '{fact[:60]}'"
    )

    # Only meaningful for NEI; None for SUPPORTS/REFUTES, where the verifier
    # reached a verdict and the quality of retrieval is not in question.
    nei_kind = (
        diagnose_nei(fact, evidence_pack["chunks"])
        if verdict.label.upper().startswith("NOT ENOUGH")
        else None
    )

    return {
        "claim"      : fact,
        "label"      : verdict.label,
        "rationale"  : verdict.rationale,
        "evidence"   : verdict.evidence,
        "nei_kind"   : nei_kind,
        "ner_query"  : evidence_pack["query"],
        "source_url" : evidence_pack["source_url"],
        # Per-stage durations; extraction_s is attached by the loop, which
        # knows how much extraction-LLM time this fact consumed.
        "timings"    : {
            "retrieval_s"    : round(retrieval_s, 2),
            "verification_s" : round(verification_s, 2),
        },
    }

def _extract_from_clauses(
    document: str,
    clauses: list[str],
    doc_words: set[str],
    seen_claims: set[str],
    source_is_fluent: bool = True,
) -> list[dict]:
    """
    One claim per clause. The loop is bounded before it starts.

    WHY THIS REPLACES THE SEARCH LOOP: the old loop asked the model "what
    relation have you not extracted yet?", which is set-difference over
    semantics and beyond phi3:mini. Two failure modes followed, both visible in
    logs:

      COVERAGE   Given a two-relation sentence it returned the first relation
                 verbatim three times and the second — the FALSE one — was
                 never checked at all.

      WASTE      Even when both relations were found, nothing knew the sentence
                 was exhausted; only the model could say so by emitting
                 "Terminate", which it does not. Every sentence therefore paid
                 3-4 extra calls, ~10-20s each, purely to discover that nothing
                 remained. That was roughly half of all extraction time.

    Syntax answers both questions for free. The clause count IS the fact count,
    so there is no discovery call and no termination signal to coax out of the
    model. Its remaining job — turn a clause into a standalone sentence — is
    one it can actually do.

    Retries still exist, but they are now scoped to a single clause rather than
    to an open-ended search, so a clause that keeps failing the gates costs at
    most MAX_CONSECUTIVE_REJECTIONS calls and does not consume the budget of
    the clauses after it.
    """
    results: list[dict] = []
    pending_extraction_s = 0.0

    print(f"[Extractor]   {len(clauses)} clause(s) to render.")

    for clause_no, clause in enumerate(clauses, 1):
        print(f"\n[Extractor]   Clause {clause_no}/{len(clauses)}: '{clause}'")
        rejected_attempts: list[tuple[str, str, str]] = []

        for attempt in range(MAX_CONSECUTIVE_REJECTIONS):
            prompt = build_extraction_prompt(
                document,
                [r["claim"] for r in results],
                [r["label"] for r in results],
                [r["rationale"] for r in results],
                rejected_attempts,
                target_clause=clause,
            )

            t_llm = time.perf_counter()
            response = call_ollama(prompt)
            pending_extraction_s += time.perf_counter() - t_llm
            print(f"[Extractor] Model response: {response}")

            fact = parse_fact_from_response(response, len(results) + 1)

            if fact == TERMINATE:
                print("[Extractor]   Clause is not a checkable statement — skipping.")
                break

            if fact is None:
                rejected_attempts.append(
                    (response.strip()[:200], "malformed", "no 'Fact_N:' line")
                )
                continue

            normalized = _normalize_claim(fact)
            if normalized in seen_claims:
                # Under clause mode a repeat means this clause restates one
                # already covered — move on rather than arguing with the model.
                print(f"[Extractor]   Clause restates an existing fact — skipping.")
                break

            if source_is_fluent:
                defect = ner.check_fluency(fact)
                if defect:
                    print(f"[Extractor]   Rejected disfluent claim — {defect}")
                    rejected_attempts.append((fact, "fluency", defect))
                    continue

            dangling = ner.check_decontextualized(fact)
            if dangling:
                print(f"[Extractor]   Rejected context-dependent claim — {dangling}")
                rejected_attempts.append((fact, "decontextualization", dangling))
                continue

            foreign = _content_words(fact) - doc_words - FUNCTION_WORDS
            if foreign:
                print(f"[Extractor]   Rejected unfaithful fact — words absent from "
                      f"the document: {sorted(foreign)} | '{fact}'")
                rejected_attempts.append((fact, "faithfulness", ", ".join(sorted(foreign))))
                continue

            unworthy = ner.check_worthy(fact)
            if unworthy:
                print(f"[Extractor]   Skipping unverifiable claim — {unworthy}")
                results.append({
                    "claim": fact, "label": "NOT ENOUGH INFO",
                    "nei_kind": "UNVERIFIABLE",
                    "rationale": f"Not sent for verification because {unworthy}.",
                    "evidence": "", "ner_query": "", "source_url": "",
                    "source_sentence": clause,
                    "timings": {"retrieval_s": 0.0, "verification_s": 0.0,
                                "extraction_s": round(pending_extraction_s, 2)},
                })
                pending_extraction_s = 0.0
                seen_claims.add(normalized)
                break

            result = grounded_verify(fact)
            result["timings"]["extraction_s"] = round(pending_extraction_s, 2)
            pending_extraction_s = 0.0
            result["source_sentence"] = clause
            results.append(result)
            seen_claims.add(normalized)

            print(f"[Extractor]   Accepted: {fact}")
            print(f"[Extractor]   Grounded feedback -> {result['label']} | "
                  f"{result['rationale']}")
            break
        else:
            print(f"[Extractor]   Clause {clause_no} failed "
                  f"{MAX_CONSECUTIVE_REJECTIONS} attempts — moving on.")

    return results


def _extract_from_sentence(
    document: str,
    sentence: str,
    doc_words: set[str],
    seen_claims: set[str],
    source_is_fluent: bool = True,
) -> list[dict]:
    """
    THE ORIGINAL SEARCH LOOP. Reached only when CLAUSE_MODE is False.

    It no longer dispatches — extract_atomic_facts chooses the path now. This
    function used to begin by testing CLAUSE_MODE and handing multi-clause
    sentences to _extract_from_clauses, which made it simultaneously the
    dispatcher and one of the two things it dispatched to. Reading it, you could
    not tell whether it was live code or a fallback, and the answer was "both,
    depending on the sentence".

    WHY IT IS NO LONGER THE DEFAULT. The loop asks the model "which relation
    have you not extracted yet?" — set-difference over semantics, which
    phi3:mini cannot do. Retained as the CLAUSE_MODE=False comparison baseline,
    because "clause enumeration beats the search loop" is a claim this project
    should be able to reproduce rather than assert.

    `document` is passed to the prompt as read-only context so references
    resolve across sentence boundaries, and `doc_words` — content words of the
    whole document, not just this sentence — is what the faithfulness gate
    checks against. That combination lets "You can walk between them" become
    "...between Africa and Europe" while still rejecting anything the document
    never says.

    `seen_claims` is shared across sentences so a fact already extracted earlier
    is not re-verified at ~150s a piece.
    """

    results: list[dict] = []

    # Extraction-LLM time accumulated since the last accepted fact —
    # rejected/malformed iterations are billed to the next accepted fact.
    pending_extraction_s = 0.0

    text_words = doc_words

    # Rejections for the CURRENT fact slot, as (attempt, foreign_words) pairs.
    # Fed back into the prompt so each retry asks a different question, and
    # cleared as soon as a fact is accepted.
    rejected_attempts: list[tuple[str, str, str]] = []   # (attempt, kind, detail)

    for iteration in range(MAX_ITERATIONS_PER_SENTENCE):
        print(f"\n[Extractor]   Iteration {iteration + 1}")

        if len(rejected_attempts) >= MAX_CONSECUTIVE_REJECTIONS:
            print(
                f"[Extractor]   {len(rejected_attempts)} consecutive rejections — "
                f"giving up on this sentence and moving to the next."
            )
            break

        # Build prompt using the full historical feedback loop
        prompt = build_extraction_prompt(
            document,
            [r["claim"] for r in results],
            [r["label"] for r in results],
            [r["rationale"] for r in results],
            rejected_attempts,
            target_sentence=sentence,
        )

        t_llm = time.perf_counter()
        response = call_ollama(prompt)
        pending_extraction_s += time.perf_counter() - t_llm
        print(f"[Extractor] Model response: {response}")

        fact = parse_fact_from_response(response, len(results) + 1)

        if fact == TERMINATE:
            print("[Extractor]   Sentence fully covered.")
            break

        if fact is None:
            # Malformed output (no strict 'Fact_N:' line, no 'Terminate').
            # Reject this iteration rather than accepting garbage as a claim;
            # the MAX_ITERATIONS ceiling still bounds the loop.
            print("[Extractor]   Rejected malformed response — retrying iteration.")
            rejected_attempts.append(
                (response.strip()[:200], "malformed", "no 'Fact_N:' line")
            )
            continue

        # Duplicate gate — STOPS AT THE FIRST REPEAT. Unlike every other
        # rejection below, this one does not retry.
        #
        # HISTORY, because this setting has now been in three states and the
        # reasoning only makes sense in order.
        #
        #   break      Original. A repeat was read as "coverage complete but the
        #              model forgot to say Terminate". That conflated two states:
        #                "nothing remains"              -> stopping is correct
        #                "I can't think of what else"   -> stopping loses a fact
        #              On "...founded in May 1607 AND is celebrated as the
        #              earliest European permanent settlement...", the extractor
        #              produced the date fact, repeated it, and the loop stopped.
        #              The "earliest" relation — the FALSE half, the entire
        #              reason for checking the sentence — was never extracted.
        #              Worse, the drop left no trace in the output.
        #
        #   reject x3  Fixed that by feeding the repeat back and granting two
        #              more attempts, so a model that had merely run out of ideas
        #              got a push.
        #
        #   break      Here. Because CLAUSE MODE removed the problem that
        #   (again)    justified retrying. This loop now runs ONLY on sentences
        #              enumerate_clauses reports as a SINGLE clause — one
        #              proposition. A duplicate in a one-proposition sentence
        #              admits exactly two readings, and retrying helps with
        #              neither:
        #
        #                exhaustion    the clause really does restate an earlier
        #                              fact; there is no second relation to find.
        #                contamination the model answered about a DIFFERENT
        #                              sentence — observed when the target opens
        #                              with an unresolved pronoun and the full
        #                              document sits above it as context. The
        #                              retry advice ("extract a different
        #                              relation") does not address the actual
        #                              error, which is that it read the wrong
        #                              sentence.
        #
        # EVIDENCE: across every logged run the duplicate rejection fires and
        # never once recovers. Three wasted calls at ~7s each, zero facts. The
        # other rejection kinds are kept at MAX_CONSECUTIVE_REJECTIONS because
        # they name a concrete, fixable defect — a malformed line, a dangling
        # pronoun, a foreign word — and those DO recover on the retry.
        #
        # Returning to `break` is only safe because extract_atomic_facts now
        # records an EXTRACTION_FAILURE for a sentence that yields nothing. What
        # made the original break dangerous was the silence, not the stopping.
        normalized = _normalize_claim(fact)
        if normalized in seen_claims:
            print(
                f"[Extractor]   Duplicate fact — stopping this sentence. Either it "
                f"restates an earlier fact or the model answered about a different "
                f"sentence; a retry fixes neither: '{fact}'"
            )
            break

        # Faithfulness gate — containment, not overlap.
        #
        # The previous version rejected a fact only when it shared ZERO content
        # words with the input, which is far too weak to catch the failure mode
        # that matters: the extractor silently *correcting* the claim. Given
        # "The Bosporus separates Africa and Europe", phi3 returned "The
        # Bosporus separates Asia Minor and Thrace" — it substituted what it
        # knows to be true for what the text actually asserts. That fact still
        # shared 'bosporus' and 'separates' with the input, so the old gate
        # passed it. Verification then checked the corrected claim, returned
        # SUPPORTS, and the pipeline lost the very error it exists to detect.
        #
        # So the requirement is now containment: every content word of the fact
        # must trace back to the source (modulo function words and inflection).
        # Any residue is material the model brought in from its own knowledge.
        # False rejections are cheap — the iteration is retried — while a false
        # acceptance corrupts the verdict, so the asymmetry favours strictness.
        # Fluency gate (Ullrich et al. 2025, §4.1) — a garbled claim produces a
        # garbled retrieval query and an uninterpretable verdict. Rejecting
        # costs one extraction call (~19s); admitting costs ~152s downstream.
        #
        # The escape hatch matters: if the SOURCE sentence is itself
        # ungrammatical (typo, missing verb), then a faithful extraction is
        # ungrammatical too. Without this, the gate would reject it, the model
        # would faithfully reproduce it, and after three rounds the sentence
        # would be silently dropped — making sloppily written input
        # uncheckable. Faithfulness outranks fluency.
        if source_is_fluent:
            defect = ner.check_fluency(fact)
            if defect:
                print(f"[Extractor]   Rejected disfluent claim — {defect} | '{fact}'")
                rejected_attempts.append((fact, "fluency", defect))
                continue

        # Decontextualization gate (AIDA 'Independent'). Runs BEFORE the
        # faithfulness gate on purpose: resolving 'them' to 'Africa and Europe'
        # pulls words from the document, which the faithfulness check then
        # validates. Reversing the order would let an unresolved claim through
        # simply because it borrowed nothing.
        dangling = ner.check_decontextualized(fact)
        if dangling:
            print(f"[Extractor]   Rejected context-dependent claim — {dangling} | '{fact}'")
            rejected_attempts.append((fact, "decontextualization", dangling))
            continue

        foreign = _content_words(fact) - text_words - FUNCTION_WORDS
        if foreign:
            print(
                f"[Extractor]   Rejected unfaithful fact — words absent from the "
                f"document: {sorted(foreign)} | '{fact}'"
            )
            rejected_attempts.append((fact, "faithfulness", ", ".join(sorted(foreign))))
            continue

        # Check-worthiness gate. Unlike the two above this does NOT retry:
        # re-extracting from an opinion yields another opinion. The claim is
        # recorded so the user sees it was considered and why, marked so it is
        # distinguishable from a claim that was checked and found wanting.
        #
        # Kept as nei_kind rather than a new top-level label, because
        # postprocessor.normalize_label() coerces anything outside
        # {SUPPORTS, REFUTES, NOT ENOUGH INFO} back to NOT ENOUGH INFO and the
        # frontend has no CSS class for a fourth value. Promoting UNVERIFIABLE
        # to a first-class label is a small change in both files — worth doing,
        # but it touches the UI, so it is not done unilaterally here.
        unworthy = ner.check_worthy(fact)
        if unworthy:
            print(f"[Extractor]   Skipping unverifiable claim — {unworthy} | '{fact}'")
            results.append({
                "claim"      : fact,
                "label"      : "NOT ENOUGH INFO",
                "nei_kind"   : "UNVERIFIABLE",
                "rationale"  : f"Not sent for verification because {unworthy}.",
                "evidence"   : "",
                "ner_query"  : "",
                "source_url" : "",
                "source_sentence": sentence,
                "timings"    : {
                    "retrieval_s"    : 0.0,
                    "verification_s" : 0.0,
                    "extraction_s"   : round(pending_extraction_s, 2),
                },
            })
            pending_extraction_s = 0.0
            seen_claims.add(normalized)
            rejected_attempts.clear()
            continue

        # Evidence-grounded verification (replaces the world-knowledge mock).
        # The verified label/rationale feed the next iteration's history AND
        # constitute the final result for this fact.
        result = grounded_verify(fact)
        result["timings"]["extraction_s"] = round(pending_extraction_s, 2)
        pending_extraction_s = 0.0
        # Provenance: which input sentence this verdict came from, so the UI
        # can highlight the offending sentence rather than the whole passage.
        result["source_sentence"] = sentence
        results.append(result)
        seen_claims.add(normalized)
        rejected_attempts.clear()   # fresh slate for the next fact slot

        print(f"[Extractor]   Accepted: {fact}")
        print(f"[Extractor]   Grounded feedback -> {result['label']} | {result['rationale']}")

    return results


def extract_atomic_facts(text: str) -> list[dict]:
    """
    Per-sentence AFEV extraction with evidence-grounded verification.

    Each sentence gets its own decomposition budget. Previously one loop ran
    over the whole document, which meant a sentence the model would not
    decompose faithfully consumed every remaining iteration: an observed run
    spent all its retries trying to restate sentence 1 as "between Asia and
    Europe" and never examined sentence 2 at all. Scoping per sentence turns
    that from a document-wide failure into a local one.

    Returns a flat list of per-fact result dicts (see grounded_verify), each
    tagged with the sentence it came from.
    """
    sentences = ner.split_sentences(text)
    if not sentences:
        return []

    doc_words = _content_words(text)
    seen_claims: set[str] = set()
    results: list[dict] = []

    for idx, sentence in enumerate(sentences, 1):
        print(f"\n[Extractor] Sentence {idx}/{len(sentences)}: '{sentence}'")

        # MAX_ITERATIONS is now a whole-document ceiling rather than the only
        # bound; without it, a long input could still run unboundedly long.
        if len(results) >= MAX_ITERATIONS:
            print("[Extractor] Document-wide fact ceiling reached — stopping.")
            break

        # If the input sentence is itself malformed, disable the fluency gate
        # for claims drawn from it — see the escape hatch note in the loop.
        source_defect = ner.check_fluency(sentence)
        if source_defect:
            print(f"[Extractor] Source sentence is not well-formed ({source_defect}); "
                  f"fluency gate disabled for it.")

        # EVERY sentence goes through clause enumeration — including sentences
        # that enumerate to exactly ONE clause.
        #
        # Previously a single-clause sentence fell back to the search loop, and
        # that fallback is where the pipeline lost a claim. The two prompts ask
        # fundamentally different questions:
        #
        #   search loop   document as CONTEXT + "TARGET SENTENCE" + "extract the
        #                 next atomic fact". The model must hold the distinction
        #                 between material it may read and material it may
        #                 extract from. phi3:mini does not.
        #
        #   clause mode   "CLAUSE TO REWRITE: <clause>. Rewrite it as one
        #                 standalone sentence, resolving pronouns from the
        #                 document." One clause is on the table. There is no
        #                 selection to get wrong.
        #
        # Observed failure of the first form: given "It is universally
        # acknowledged as longer than the Nile by all international
        # cartographers" as the target, the extractor answered with a fact from
        # the PREVIOUS sentence. The target opens with an unresolved pronoun and
        # the full document sits above it, so the most salient extractable thing
        # in the prompt was not in the target at all. Three duplicate rejections
        # later the sentence — the false claim in the input, and the entire
        # reason the request was made — was dropped.
        #
        # enumerate_clauses already returns [sentence] for a single-clause
        # sentence, so no special case is needed here: the clause list is simply
        # of length one and the same code runs.
        #
        # This is a routing change, not a new mechanism. The single-clause path
        # now uses a prompt that was already written, already used, and already
        # working for multi-clause sentences.
        if CLAUSE_MODE:
            clauses = ner.enumerate_clauses(sentence)
            sentence_results = _extract_from_clauses(
                text, clauses, doc_words, seen_claims,
                source_is_fluent=source_defect is None,
            )
        else:
            sentence_results = _extract_from_sentence(
                text, sentence, doc_words, seen_claims,
                source_is_fluent=source_defect is None,
            )

        print(f"[Extractor] Sentence {idx} yielded {len(sentence_results)} fact(s).")

        # A SENTENCE THAT YIELDS NOTHING IS A FAILURE, NOT AN ABSENCE.
        #
        # Until now this branch did not exist: extraction printed "yielded 0
        # fact(s)", moved on, and the request reported an overall verdict
        # computed from whatever the other sentences produced. In the output
        # there was no difference at all between
        #
        #     "this sentence contains no checkable claim"     — fine
        #     "extraction broke and the claim was lost"       — the founding bug
        #
        # An observed run: sentence 2 was "It is universally acknowledged as
        # longer than the Nile by all international cartographers" — the FALSE
        # claim in the input, and the whole reason the request was made. The
        # extractor answered with a fact from sentence 1 (the target sentence
        # opens with an unresolved 'It', and the full document sits above it in
        # the prompt as context), the duplicate gate correctly rejected it three
        # times, and the sentence was dropped in silence. The request returned
        # REFUTES on the OTHER sentence and never mentioned the loss.
        #
        # This does not fix the extraction failure — that is the clause-routing
        # change, and it is a separate run. It makes the failure VISIBLE, which
        # has to come first: a bug that leaves no trace in the output cannot be
        # measured, and a fix for it cannot be evaluated.
        #
        # Labelled NOT ENOUGH INFO because postprocessor.normalize_label()
        # coerces anything outside the three FEVER labels back to it anyway;
        # nei_kind carries the real distinction, exactly as UNVERIFIABLE does.
        if not sentence_results:
            print(
                f"[Extractor] EXTRACTION FAILURE on sentence {idx} — no claim was "
                f"produced. This is recorded, not discarded: the sentence may "
                f"have carried an assertion that never reached verification."
            )
            results.append({
                "claim"      : sentence,
                "label"      : "NOT ENOUGH INFO",
                "nei_kind"   : "EXTRACTION_FAILURE",
                "rationale"  : (
                    "No atomic claim could be extracted from this sentence, so it "
                    "was never verified. The verdict describes the extractor, not "
                    "the sentence — treat this as unchecked, not as unsupported."
                ),
                "evidence"   : "",
                "ner_query"  : "",
                "source_url" : "",
                "source_sentence": sentence,
                "timings"    : {
                    "retrieval_s"    : 0.0,
                    "verification_s" : 0.0,
                    "extraction_s"   : 0.0,
                },
            })
            continue

        results.extend(sentence_results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(text: str) -> list[dict]:
    """
    Full claim extraction + verification pipeline:
    precondition → iterative extraction with evidence-grounded feedback
    → list of verified per-fact result dicts for postprocessor.process().
    """
    clean_text = precondition(text)
    if clean_text is None:
        return []

    results = extract_atomic_facts(clean_text)

    print(f"\n[Extractor] Done. {len(results)} verified atomic fact(s):")
    for i, r in enumerate(results, 1):
        suffix = ""
        if r.get("nei_kind") == "RETRIEVAL_FAILURE":
            suffix = "  <- evidence never mentioned the subject; verdict not reached"
        elif r.get("nei_kind") == "GENUINE":
            suffix = "  <- relevant evidence retrieved but inconclusive"
        elif r.get("nei_kind") == "UNVERIFIABLE":
            suffix = "  <- not check-worthy; never sent for verification"
        elif r.get("nei_kind") == "EXTRACTION_FAILURE":
            suffix = "  <- EXTRACTION FAILED; sentence never became a claim"
        print(f"  {i}. [{r['label']}] {r['claim']}{suffix}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(text_ba)