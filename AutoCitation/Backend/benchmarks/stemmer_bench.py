"""
stemmer_bench.py — which stemmer should the faithfulness gate use?

This is an EVALUATION, not a test. It prints numbers to compare; it has no
pass/fail. Binary invariants that must never regress live in
Tests/claim_extractor_test.py instead.

    cd Backend
    python benchmarks/stemmer_bench.py
    python benchmarks/stemmer_bench.py --verbose        # every disagreement
    python benchmarks/stemmer_bench.py --only spacy     # substring filter

Evaluation data lives in stemmer_data.py.

WHY THE METRIC IS SHAPED THIS WAY
---------------------------------
A stemmer is not good or bad in the abstract, only relative to the job. The
gate computes:

    foreign = content_words(claim) - content_words(document) - FUNCTION_WORDS

and rejects the claim when `foreign` is non-empty. Two failure directions,
with very different costs:

    too weak       'separated' != 'separates'  -> faithful claim REJECTED
                   costs ~19s of re-extraction

    too aggressive 'organization' -> 'organ'   -> unfaithful claim ACCEPTED
                   costs a corrupted verdict

The second is the failure the gate exists to prevent, so FALSE ACCEPTANCES are
read first and false rejections are the tiebreak. There is deliberately no
single blended accuracy figure: it would hide exactly this asymmetry.

CANDIDATES TAKE OPTIONAL CONTEXT
--------------------------------
Every candidate has the signature f(word, sentence=None) -> str. Most ignore
the sentence. The propn-skip variant cannot work without it: given the bare
token "Wales", spaCy has no way to know it is a country rather than the plural
of 'wale', so on the first version of this benchmark that candidate scored
identically to plain lemmatisation and the comparison was meaningless.
"""

import argparse
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from Pipeline import claim_extractor as ce          # noqa: E402
from benchmarks import stemmer_data as data         # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Candidates
# ═════════════════════════════════════════════════════════════════════════════
def _build_candidates() -> dict:
    """Every stemmer that loads here. Missing ones are skipped, not fatal."""
    candidates = {"suffix_stem": lambda w, s=None: ce._suffix_stem(w)}

    try:
        from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer
        sb, pt, lc = SnowballStemmer("english"), PorterStemmer(), LancasterStemmer()
        candidates["snowball"] = lambda w, s=None: sb.stem(w)
        candidates["porter"] = lambda w, s=None: pt.stem(w)
        # DELIBERATE CONTROL. Lancaster is famously over-aggressive and should
        # score worst on B, C and FA. If it does not, this benchmark is not
        # detecting over-stemming and no other row can be trusted.
        candidates["lancaster (control)"] = lambda w, s=None: lc.stem(w)
    except ImportError:
        print("  [skip] nltk not installed — snowball/porter/lancaster omitted")

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

        def spacy_lemma(word, sentence=None):
            return nlp(word)[0].lemma_.lower()

        def spacy_lemma_keep_propn(word, sentence=None):
            """
            Lemmatise common words; leave proper nouns exactly as written.

            No pure stemmer can pass 'Paris' and 'parries' differently once the
            string is lowercased — the information is gone. spaCy tags PROPN,
            but only with enough context to decide, which is why the sentence
            argument exists and why set C stores one per name.
            """
            if sentence:
                doc = nlp(sentence)
                for tok in doc:
                    if tok.text.lower() == word.lower():
                        return (word.lower() if tok.pos_ == "PROPN"
                                else tok.lemma_.lower())
            return nlp(word)[0].lemma_.lower()

        candidates["spacy_lemma"] = spacy_lemma
        candidates["spacy_lemma + propn skip"] = spacy_lemma_keep_propn
    except Exception as e:
        print(f"  [skip] spaCy unavailable ({type(e).__name__}) — lemma rows omitted")

    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════════════════════
def _content_words_with(stem, text: str) -> set[str]:
    """
    _content_words with a swappable stemmer.

    The whole sentence is passed as context for every token, so the propn-skip
    candidate can see which words are names — this mirrors what the real
    function would have to do to adopt that strategy.
    """
    words = re.findall(r"[a-zçğışöü0-9]+", text.lower())
    return {
        stem(w, text) for w in words
        if len(w) >= ce.MIN_CONTENT_WORD_LENGTH
    }


def score(name: str, stem, verbose: bool) -> dict:
    detail = []

    collapse_ok = 0
    for a, b in data.MUST_COLLAPSE:
        if stem(a) == stem(b):
            collapse_ok += 1
        elif verbose:
            detail.append(f"    A miss:        {a}->{stem(a)}  vs  {b}->{stem(b)}")

    distinct_ok = 0
    for a, b in data.MUST_NOT_COLLAPSE:
        if stem(a) != stem(b):
            distinct_ok += 1
        elif verbose:
            detail.append(f"    B conflation:  {a} + {b} -> {stem(a)}")

    propn_ok = 0
    for word, sentence in data.PROPER_NOUNS:
        if stem(word, sentence) == word:
            propn_ok += 1
        elif verbose:
            detail.append(f"    C mangled:     {word} -> {stem(word, sentence)}")

    false_accept = false_reject = 0
    for case in data.GATE_CASES:
        expected = case["expected_foreign"]
        # The negation case is unreachable for every candidate. It stays in the
        # data so the limitation is visible, and out of the counts so it does
        # not penalise everyone equally and tell you nothing.
        if "<negation>" in expected:
            continue

        got = (_content_words_with(stem, case["claim"])
               - _content_words_with(stem, case["document"])
               - ce.FUNCTION_WORDS)
        expected_stemmed = {stem(w) for w in expected}

        if expected_stemmed and not got:
            false_accept += 1
            detail.append(f"    FALSE ACCEPT:  {case['note']}")
        elif not expected_stemmed and got:
            false_reject += 1
            detail.append(f"    false reject:  {case['note']} — flagged {sorted(got)}")

    return {
        "name": name,
        "collapse": collapse_ok, "collapse_n": len(data.MUST_COLLAPSE),
        "distinct": distinct_ok, "distinct_n": len(data.MUST_NOT_COLLAPSE),
        "propn": propn_ok, "propn_n": len(data.PROPER_NOUNS),
        "fa": false_accept, "fr": false_reject,
        "detail": detail,
    }


def _pct(n, d):
    return f"{n:>3}/{d:<3} {n / d * 100:>5.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list every disagreement")
    ap.add_argument("--only", default="", help="substring filter on candidate name")
    args = ap.parse_args()

    print("\nLoading candidates...")
    candidates = _build_candidates()
    if args.only:
        candidates = {k: v for k, v in candidates.items() if args.only.lower() in k.lower()}
    print(f"  {len(candidates)} candidate(s): {', '.join(candidates)}")
    print(f"  sets: A={len(data.MUST_COLLAPSE)}  B={len(data.MUST_NOT_COLLAPSE)}  "
          f"C={len(data.PROPER_NOUNS)}  D={len(data.GATE_CASES)}\n")

    rows = [score(n, s, args.verbose) for n, s in candidates.items()]

    header = (f"{'stemmer':26} {'A collapse':>13} {'B distinct':>13} "
              f"{'C propn':>13} {'FA':>4} {'FR':>4}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:26} "
              f"{_pct(r['collapse'], r['collapse_n']):>13} "
              f"{_pct(r['distinct'], r['distinct_n']):>13} "
              f"{_pct(r['propn'], r['propn_n']):>13} "
              f"{r['fa']:>4} {r['fr']:>4}")

    if args.verbose:
        for r in rows:
            if r["detail"]:
                print(f"\n  {r['name']}")
                print("\n".join(r["detail"]))

    print(f"""
READING THE TABLE

  FA  false acceptances — an unfaithful claim the gate let through.
      READ THIS FIRST. Each one is a corrupted verdict.
  FR  false rejections — a faithful claim wrongly refused, ~19s each.
      Tiebreak only.

  A   inflections that must collapse    low -> more FR
  B   distinct words that must not      low -> more FA
  C   proper nouns left intact          fact-checking is full of them

  Sanity check before believing any row: 'lancaster (control)' must score
  worst on B, C and FA. If it does not, the benchmark is not measuring
  over-stemming.

  Current setting: STEMMER_LANGUAGE={ce.STEMMER_LANGUAGE!r}  "
  USE_SNOWBALL_STEMMER={ce.USE_SNOWBALL_STEMMER}
""")


if __name__ == "__main__":
    main()
