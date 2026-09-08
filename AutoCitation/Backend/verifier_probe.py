"""
Single-call verifier probe.

WHY THIS EXISTS
---------------
Three consecutive prompt edits failed to move one verdict, and each attempt
cost a full pipeline run — 400-1000 seconds of extraction, Wikipedia retrieval
and clause enumeration — to test a change that affects exactly one inference
call. That is one experiment per sitting, and it is why the last three changes
were guesses rather than measurements.

This runs that one call and nothing else. Claim in, evidence in, verdict out,
in roughly the time qwen3 takes to think (~90-240s on CPU). No spaCy, no
network, no extraction. Decoding is greedy with a fixed seed, so a repeat run
with an unchanged prompt is bit-identical: any difference you see is caused by
the thing you changed.

It also prints the <think> block, which the pipeline throws away. That block is
the only direct evidence of WHY a label was chosen — whether the model read a
rule, which words it fixated on, whether it silently reworded the claim. The
Amazon over-refutation was diagnosed from a leaked fragment of exactly this
kind ("...the claim that the Amazon holds the title UNAMBIGUOUSLY", where
'unambiguously' was nowhere in the claim).

USAGE
-----
    python verifier_probe.py --list
    python verifier_probe.py --case amazon_disputed
    python verifier_probe.py --all
    python verifier_probe.py --all --sweep-thinking
    python verifier_probe.py --case amazon_disputed --show-prompt --show-think
    python verifier_probe.py --claim "..." --evidence "..." --evidence "..."

WHAT IT DOES NOT TEST
---------------------
Retrieval. Every case here hands the verifier a chunk chosen by hand, which is
the point — it separates "the verifier misread good evidence" from "retrieval
never delivered the evidence". Those two failures look identical in a pipeline
log and need opposite fixes. Do not read a passing probe as a passing pipeline.
"""

import argparse
import re
import sys
import time

import requests

import Pipeline.verifier as verifier


# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────
# Each case pairs a claim with evidence copied VERBATIM from a real run, plus
# the label that evidence actually supports.
#
# `expect` is what the chunk justifies, NOT what you wish were true about the
# world. amazon_disputed expects NOT ENOUGH INFO rather than REFUTES because
# "disputed with the Nile" reports an open question — refuting the claim would
# need a source that settles it, and no such chunk was retrieved.
#
# THE CONTROLS ARE NOT OPTIONAL. jamestown_date and jamestown_augustine exist
# to catch the failure mode a rule change is most likely to cause: a prompt
# that fixes over-refutation by making the model answer NOT ENOUGH INFO to
# everything scores well on the two problem cases and has destroyed the system.
# Always read all four.
CASES: dict[str, dict] = {
    "amazon_disputed": {
        "why": "Over-refutation. The model added 'unambiguously' to the claim, "
               "then refuted its own addition. Evidence reports an OPEN question.",
        "claim": "The title of the world's longest river belongs to the Amazon River.",
        "evidence": [
            "The Amazon River in South America is the largest river in the world by "
            "discharge volume of water, and the second-longest or longest river system "
            "in the world, a title which is disputed with the Nile. For nearly a century, "
            "the headwaters of the Apurimac River on Nevado Mismi in Peru had been "
            "considered the Amazon basin's most distant source.",
        ],
        "expect": "NOT ENOUGH INFO",
    },

    "jamestown_nationality": {
        "why": "Partial comparison. Evidence says ENGLISH, claim says EUROPEAN. The "
               "model checked the geographic scope and skipped the nationality scope.",
        "claim": "Jamestown is celebrated as the earliest European permanent settlement "
                 "in what is now the United States.",
        "evidence": [
            "The Jamestown settlement in the Colony of Virginia was the first permanent "
            "English settlement in the Americas. It was established by the Virginia "
            "Company of London as \"James Fort\" on May 4, 1607 O.S. (May 14, 1607 N.S.). "
            "It followed earlier, failed English colonization attempts, including the "
            "1585 Roanoke Colony.",
        ],
        "expect": "NOT ENOUGH INFO",
    },

    "jamestown_augustine": {
        "why": "CONTROL — can the verifier refute at all? Given the chunk retrieval "
               "usually misses, REFUTES is the only defensible answer.",
        "claim": "Jamestown is celebrated as the earliest European permanent settlement "
                 "in what is now the United States.",
        "evidence": [
            "Native Americans lived in the territory of North America prior to European "
            "colonization and were present throughout the colonial era. The Spanish were "
            "the first Europeans to establish a permanent settlement in what became the "
            "United States, at Saint Augustine, Florida (1565).",
        ],
        "expect": "REFUTES",
    },

    "jamestown_date": {
        "why": "CONTROL — a narrower statement that DOES entail the broader claim. "
               "A specific date establishes the month. Guards against a rule that "
               "turns every narrowing into NOT ENOUGH INFO.",
        "claim": "The English settlement of Jamestown was founded in May 1607.",
        "evidence": [
            "Historic Jamestowne is established in the original James Fort and Jamestown "
            "Colony, the first successful English settlement on the mainland of North "
            "America, founded on May 14, 1607.",
        ],
        "expect": "SUPPORTS",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Raw call — same prompt, same options, reasoning NOT stripped
# ─────────────────────────────────────────────────────────────────────────────
def call_raw(prompt: str) -> tuple[str, str, float]:
    """
    Post the prompt to Ollama; return (answer, reasoning, seconds).

    Every option is imported from verifier rather than restated, so this cannot
    drift out of sync with what the pipeline actually sends. The ONE difference
    from verifier.call_ollama is deliberate: it does not strip the reasoning
    block. That block is the whole reason this probe is useful.
    """
    payload = {
        "model": verifier.REASONING_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": verifier.VERIFIER_THINKING,
        "keep_alive": verifier.KEEP_ALIVE,
        "options": {
            "num_ctx": verifier.VERIFIER_NUM_CTX,
            "num_predict": verifier.VERIFIER_NUM_PREDICT,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": 0,
        },
    }
    t0 = time.perf_counter()
    response = requests.post(verifier.OLLAMA_URL, json=payload)
    response.raise_for_status()
    elapsed = time.perf_counter() - t0
    body = response.json()

    tokens = body.get("eval_count", 0)
    rate = tokens / elapsed if elapsed else 0.0
    print(f"    [{verifier.REASONING_MODEL}]  {tokens} tok in {elapsed:.1f}s ({rate:.1f} tok/s)")

    # REASONING ARRIVES IN ITS OWN FIELD, not inline in `response`.
    #
    # When `think` is set, Ollama separates the reasoning into body["thinking"]
    # and returns only the answer in body["response"]. The first version of this
    # probe searched `response` for a <think> block, found none, and printed
    # "(no <think> block)" on a run that had just spent 2063 tokens reasoning.
    #
    # verifier.call_ollama's regex strip is therefore belt-and-braces rather
    # than the mechanism that removes it — correct to keep (older Ollama builds
    # and think=True on some models do inline it) but it is not what makes the
    # pipeline's reasoning invisible. The separate field is.
    return body.get("response", ""), body.get("thinking", "") or "", elapsed


def set_thinking(value) -> None:
    """
    Override the reasoning budget for this process.

    VERIFIER_NUM_PREDICT is computed at import time from VERIFIER_THINKING, so
    changing one without the other leaves a 256-token ceiling in front of a
    model that has just been told to reason — which truncates mid-<think> and
    returns an empty response. Both move together, always.
    """
    verifier.VERIFIER_THINKING = value
    verifier.VERIFIER_NUM_PREDICT = 256 if value is False else 3072


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_case(name: str, case: dict, args) -> bool:
    print("\n" + "=" * 78)
    print(f"CASE  {name}    thinking={verifier.VERIFIER_THINKING!r}")
    print(f"WHY   {case['why']}")
    print("=" * 78)
    print(f"CLAIM   {case['claim']}")
    for i, chunk in enumerate(case["evidence"], 1):
        print(f"EV_{i}    {chunk[:150]}{'...' if len(chunk) > 150 else ''}")

    prompt = verifier.build_verification_prompt(case["claim"], case["evidence"])

    if args.show_prompt:
        print("\n--- PROMPT ---\n" + prompt + "\n--- END PROMPT ---")

    raw, thinking, _ = call_raw(prompt)

    # Either transport: the dedicated field, or an inline block in the answer.
    inline = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    reasoning = thinking.strip() or (inline.group(1).strip() if inline else "")

    if args.show_think:
        print("\n--- REASONING ---")
        print(reasoning or "(model returned no reasoning — is `think` set?)")
        print("--- END REASONING ---")
    elif reasoning:
        print(f"    (reasoning: {len(reasoning.split())} words — --show-think to read it)")

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    result = verifier.parse_verification_response(cleaned, case["evidence"])

    expected = case["expect"]
    ok = result.label == expected

    print(f"\n  EXPECTED  {expected}")
    print(f"  GOT       {result.label}   {'PASS' if ok else '<<<< FAIL'}")
    print(f"  RATIONALE {result.rationale}")

    # The rationale is where a wrong verdict explains itself. When the label is
    # wrong, read it before touching the prompt: the Amazon case was solved by
    # noticing a word in the rationale that was not in the claim.
    if not ok:
        claim_words = set(re.findall(r"[a-z]+", case["claim"].lower()))
        rationale_words = re.findall(r"[a-z]+", result.rationale.lower())
        added = [w for w in ("unambiguously", "universally", "undisputed", "certainly",
                             "definitively", "english", "european", "always", "never")
                 if w in rationale_words and w not in claim_words]
        if added:
            print(f"  NOTE      the rationale uses words absent from the claim: {added}")
            print("            the model may be judging a claim it reworded first.")

    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Probe verifier.verify() with one inference call.")
    p.add_argument("--case", action="append", help="case name (repeatable)")
    p.add_argument("--all", action="store_true", help="run every built-in case")
    p.add_argument("--list", action="store_true", help="list cases and exit")
    p.add_argument("--claim", help="ad-hoc claim")
    p.add_argument("--evidence", action="append", help="ad-hoc evidence chunk (repeatable)")
    p.add_argument("--expect", default="NOT ENOUGH INFO", help="expected label for --claim")
    p.add_argument("--think", choices=["false", "low", "true"],
                   help="override VERIFIER_THINKING for this run")
    p.add_argument("--sweep-thinking", action="store_true",
                   help="run each case at false, low and true")
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--show-think", action="store_true")
    args = p.parse_args()

    if args.list:
        for name, case in CASES.items():
            print(f"{name:24} expect {case['expect']:16} {case['why']}")
        return 0

    if args.claim:
        if not args.evidence:
            print("--claim requires at least one --evidence")
            return 2
        cases = {"adhoc": {"why": "ad-hoc", "claim": args.claim,
                           "evidence": args.evidence, "expect": args.expect.upper()}}
    elif args.case:
        missing = [c for c in args.case if c not in CASES]
        if missing:
            print(f"unknown case(s): {missing}. --list to see them.")
            return 2
        cases = {c: CASES[c] for c in args.case}
    elif args.all:
        cases = CASES
    else:
        p.print_help()
        return 2

    if args.think:
        set_thinking({"false": False, "low": "low", "true": True}[args.think])

    budgets = [False, "low", True] if args.sweep_thinking else [verifier.VERIFIER_THINKING]

    results: list[tuple[str, object, bool]] = []
    for budget in budgets:
        set_thinking(budget)
        for name, case in cases.items():
            results.append((name, budget, run_case(name, case, args)))

    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, budget, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24} thinking={budget!r}")
    passed = sum(1 for _, _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} passed")
    print("=" * 78)

    # Non-zero exit on failure so this can gate a commit later without changes.
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
