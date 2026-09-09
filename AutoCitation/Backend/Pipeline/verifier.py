import re
from dataclasses import dataclass

import requests

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
REASONING_MODEL = "qwen3:8b"

# ── Result type ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VerificationResult:
    """
    A verdict, the chunk that produced it, and the one-line justification.

    Deliberately NOT a tuple or NamedTuple. This module used to return three
    bare strings, and the two functions that did so disagreed about the order:
    parse_verification_response gave (label, evidence, rationale) while verify
    gave (label, rationale, evidence). Positional unpacking cannot catch that
    mistake — all three fields are strings, so a swap raises nothing and simply
    puts a whole Wikipedia paragraph where a one-sentence rationale belongs,
    surfacing much later as a puzzling frontend bug.

    Attribute access removes the ordering from the interface entirely:
    result.evidence cannot be confused with result.rationale at any call site.
    Making it a NamedTuple would have preserved unpacking, and with it the
    hazard, so unpacking is left deliberately unavailable — stale positional
    code fails loudly instead of quietly.

    frozen=True because a verdict is a record of what happened, not a mutable
    working value.
    """
    label: str
    evidence: str
    rationale: str


# ── Constants ─────────────────────────────────────────────────────────────────
VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

# qwen3 is a hybrid reasoning model: left to itself it emits a <think> block
# before its answer. call_ollama() has always stripped that block — meaning the
# tokens were generated, paid for at ~137s/claim, and then thrown away.
#
# Ollama exposes `think` as a top-level request field, so the reasoning can be
# switched off (False) or budgeted ("low"/"medium"/"high"). Discarded reasoning
# is pure waste, but it is NOT free to remove: the REFUTES verdict on "Bosporus
# is located between Africa and Europe" required inferring that naming Asia and
# Europe as the two banks excludes Africa, which is exactly the kind of implicit
# step reasoning tokens buy. So this is a knob to MEASURE, not a settled choice:
# run the regression set at False, "low", and True and compare verdicts, not
# just wall-clock.
VERIFIER_THINKING = "low"

# Evidence chunks are long Wikipedia passages; the previous call sent no
# num_ctx at all and silently inherited Ollama's default window, which risks
# truncating the very evidence the verdict depends on.
VERIFIER_NUM_CTX = 8192

# Output ceiling. With thinking disabled the answer is three short lines, so a
# tight cap costs nothing and bounds a runaway generation. With thinking on the
# budget must ALSO cover the <think> block — and 1024 was too tight.
#
# MEASURED, not guessed. Two observed runs:
#
#   [Inference] qwen3:8b  output 1024 tok in 263.7s
#   [Verifier] Raw model response:            <- empty
#
#   [Inference] qwen3:8b  output 2063 tok in 370.9s   (verifier_probe, same claim)
#
# The first is exactly at the ceiling: cut off mid-reasoning, 264 seconds spent,
# nothing returned. The strip regex then found an unterminated <think> and left
# an empty string, and parse_verification_response fell back to its default —
# so the pipeline printed
#
#   [Verifier] Label     : NOT ENOUGH INFO
#   [Verifier] Rationale :                    <- blank
#
# A CRASH THAT READS AS A JUDGEMENT. In the final report that row is
# indistinguishable from "the evidence was genuinely inconclusive", which is the
# single most misleading thing this pipeline can output: it is not a weak
# verdict, it is no verdict at all.
#
# The probe measurement says the hard cases want ~2000 tokens, so 1024 does not
# merely clip the tail — it truncates the reasoning that decides the answer. Set
# to 3072 for headroom, and verify() now DETECTS the case explicitly rather than
# letting it masquerade as a label.
VERIFIER_NUM_PREDICT = 256 if VERIFIER_THINKING is False else 3072


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Call Ollama (reasoning model)
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """Send a prompt to the reasoning model and return the raw response string."""
    payload = {
        "model": REASONING_MODEL,
        "prompt": prompt,
        "stream": False,
        # Top-level field, NOT an option — see Ollama's /api/generate schema.
        "think": VERIFIER_THINKING,
        "options": {
            "num_ctx": VERIFIER_NUM_CTX,
            "num_predict": VERIFIER_NUM_PREDICT,
            # Greedy decoding. Verification is a classification with one right
            # answer; sampling only adds a chance of picking the second-best
            # label. Ollama defaults to temperature 0.8, which this call was
            # silently inheriting.
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": 0,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    raw = response.json()["response"]

    # qwen3 emits <think>...</think> reasoning blocks by default. Strip them
    # before parsing so leaked chain-of-thought can never be mistaken for
    # the LABEL / EVIDENCE / RATIONALE fields.
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build verification prompt  (mirrors Figure 4 in the AFEV paper)
# ─────────────────────────────────────────────────────────────────────────────
def build_verification_prompt(fact: str, evidence_chunks: list[str]) -> str:
    """
    Builds the reasoning prompt based on AFEV Figure 4.
    Presents the fact and all evidence chunks to the reasoning model.
    Instructs it to select the most relevant chunk and judge the fact against it.

    RULE 6 IS THE OVER-REFUTATION FIX. Observed:

        claim    "The title of the world's longest river belongs to the Amazon."
        evidence "...the second-longest or longest river system in the world,
                  a title which is disputed with the Nile."
        verdict  REFUTES
        reason   "...directly contradicting the claim that the Amazon holds the
                  title UNAMBIGUOUSLY."

    'Unambiguously' is not in the claim. The model supplied a strength qualifier
    the claim never asserted and then refuted its own addition. That mechanism
    is also why the previous wording — "only output REFUTES when a chunk
    explicitly and unambiguously contradicts the claim" — did not prevent it:
    the model believed it HAD found an unambiguous contradiction, of a claim it
    had silently strengthened first.

    So the rule now constrains two things instead of one: what counts as a
    contradiction (something that cannot hold simultaneously, not something that
    declines to confirm), and what the claim is permitted to mean (exactly what
    it says). Evidence that reports a question as OPEN is named explicitly,
    because 'disputed' is the textbook NOT ENOUGH INFO signal and nothing in
    this prompt had ever said so — rule 5 covers chunks contradicting EACH
    OTHER, not a single chunk reporting an open question.

    VALIDATED BY verifier_probe.py before shipping, on four cases: the disputed
    river (now NOT ENOUGH INFO, was REFUTES) plus two controls that must not
    move — a claim the evidence genuinely refutes, and a narrower statement that
    legitimately entails a broader one ("founded on May 14, 1607" does establish
    "founded in May 1607"). A rule that fixes over-refutation by answering NOT
    ENOUGH INFO to everything would pass the first case and destroy the system;
    the controls are what catch that.

    THE SUPPORTS CLAUSE EXISTS BECAUSE THE FIRST VERSION LEANED ON THE SCALE.
    Every clause of rule 6 as first written pushed AWAY FROM REFUTES, and
    nothing in it pushed away from SUPPORTS. It fixed the Amazon over-refutation
    and, on the next run, a claim that had correctly come back NOT ENOUGH INFO
    on identical evidence turned into SUPPORTS:

        claim    "...the earliest EUROPEAN permanent settlement in what is now
                  the United States"
        evidence "...the first permanent ENGLISH settlement in the Americas"
        reason   "...which includes the United States. This supports the claim."

    English settlements are SOME European settlements, so being first among them
    says nothing about being first among all of them — an earlier Spanish or
    French settlement is not ruled out by that sentence. The model checked the
    geographic scope (Americas ⊇ United States, correct) and treated the
    narrower nationality as a detail rather than as the thing at issue.

    A rule that only restrains one label does not make the verifier careful, it
    makes it biased. Both directions now carry the same bar: evidence about a
    narrower group settles nothing about a wider one, whichever way it points.

    Kept as sub-clauses of rule 6 rather than as rules 7 and 8. An earlier rule 7
    on qualifier checking was withdrawn partly because seven unordered
    instructions competed for attention, and piling more on would repeat that.

    WHAT THIS CANNOT FIX, stated so the next reader does not try: with only
    Jamestown-centric evidence in the prompt, NOT ENOUGH INFO is the BEST
    available answer, not the correct one. Refuting the claim needs the passage
    about Saint Augustine (1565), and retrieval cannot find it — the query is
    built from the claim's entities, and a passage about a different settlement
    does not contain the claim's subject. That is a retrieval problem and no
    verifier prompt reaches it.

    THE WORKED EXAMPLE MUST NOT COME FROM THE TEST SET. Marlow Tower and Kessler
    Building are invented for this purpose. Using a project test sentence would
    show the model the answer to a question it is about to be asked, and that
    case would then measure nothing.
    """
    chunks_block = ""
    for i, chunk in enumerate(evidence_chunks, 1):
        chunks_block += f"Evidence_{i}: {chunk}\n"

    prompt = f"""You are an expert fact verifier. Your task is to verify the following atomic claim using only the provided evidence chunks.

Atomic Claim:
\"{fact}\"

Available Evidence:
{chunks_block.strip()}

Instructions:
1. Read all evidence chunks carefully.
2. Identify which single evidence chunk is most relevant to verifying the claim.
3. Based strictly on that chunk, determine the verification label.
4. Do not use your internal knowledge — base your judgment solely on the evidence provided.
5. If the evidence chunks contradict each other on the point in question, or none of them directly addresses the claim's subject, output NOT ENOUGH INFO rather than guessing.
6. REFUTES REQUIRES A CONTRADICTION, NOT AN ABSENCE OF CONFIRMATION. Output
   REFUTES only when a chunk states something that CANNOT BE TRUE AT THE SAME
   TIME as the claim.
   - Judge the claim exactly as written. Do not read extra strength into it. A
     plain assertion does not also assert that it is certain, undisputed, or
     universally agreed, so you may not refute it for failing to be those.
   - If the chunk presents the point as OPEN — disputed, contested, "X or Y",
     estimates vary, some sources say, widely believed — then it neither
     establishes nor contradicts the claim, and the answer is NOT ENOUGH INFO.

   Claim:    "The tallest building in the region is the Marlow Tower."
   Evidence: "The Marlow Tower is the second-tallest or tallest in the region,
              a distinction disputed with the Kessler Building."
   CORRECT   NOT ENOUGH INFO — the evidence reports the question as open.
   WRONG     REFUTES — nothing there states the Marlow Tower is not tallest.

   THE SAME BAR APPLIES TO SUPPORTS. Being FIRST, LARGEST or OLDEST within a
   NARROWER group does not establish being first, largest or oldest within a
   WIDER one. If the claim's group is wider than the evidence's group, the
   evidence leaves the wider question untouched, and the answer is NOT ENOUGH
   INFO — no matter how closely the two sentences otherwise match.

   Claim:    "The Marlow Tower was the first stone building in the region."
   Evidence: "The Marlow Tower was the first GRANITE building in the region."
   CORRECT   NOT ENOUGH INFO — granite buildings are only some stone buildings,
             so an earlier stone building of another kind is not ruled out.
   WRONG     SUPPORTS — granite is a stone, therefore the first granite
             building is the first stone building. It does not follow.

Output strictly in this format with no extra text:
LABEL: <SUPPORTS|REFUTES|NOT ENOUGH INFO>
EVIDENCE: <the number of the chosen chunk only, for example: 2>
RATIONALE: <one sentence explaining why the label was assigned based on the evidence>

Do NOT copy the evidence text. Write only its number."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parse model output
# ─────────────────────────────────────────────────────────────────────────────
def parse_verification_response(
    response: str,
    evidence_chunks: list[str]
) -> VerificationResult:
    """
    Parses the three structured fields from the reasoning model response.
    Falls back gracefully if any field is missing or malformed.

    Returns a VerificationResult; read its fields by name.
    """
    # Parse label
    label_match = re.search(
        r'LABEL:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)',
        response,
        re.IGNORECASE
    )
    label = label_match.group(1).upper() if label_match else "NOT ENOUGH INFO"

    if label not in VALID_LABELS:
        label = "NOT ENOUGH INFO"

    # Parse evidence — now an INDEX, not the chunk text.
    #
    # The model used to be asked to "copy the single most relevant evidence
    # chunk exactly", so every verdict cost ~120 tokens of verbatim Wikipedia
    # prose. Generation is sequential and dominates inference time, while the
    # chunk text is already sitting in evidence_chunks — the model was being
    # paid to retype data we already had. It now writes a number and Python
    # does the lookup.
    evidence = ""
    evidence_match = re.search(
        r'EVIDENCE:\s*(.+?)(?=\n\s*RATIONALE:|\Z)',
        response,
        re.IGNORECASE | re.DOTALL
    )

    if evidence_match:
        raw = evidence_match.group(1).strip()

        # Accept "2", "Evidence_2", "Evidence 2:", "chunk 2" — the first
        # integer in the field is the selection.
        idx_match = re.search(r'\d+', raw)
        if idx_match:
            idx = int(idx_match.group()) - 1        # prompt is 1-based
            if 0 <= idx < len(evidence_chunks):
                evidence = evidence_chunks[idx]
            else:
                print(f"[Verifier] Evidence index {idx + 1} out of range "
                      f"(1..{len(evidence_chunks)}) — falling back to top chunk.")
        elif raw:
            # Backward compatibility: a model that ignores the instruction and
            # pastes the chunk anyway should still produce a usable result.
            print("[Verifier] Model returned evidence text instead of an index.")
            evidence = raw

    if not evidence:
        # Never index into a possibly-empty chunk list — the old fallback
        # (evidence_chunks[0]) raised IndexError when retrieval found nothing.
        evidence = evidence_chunks[0] if evidence_chunks else ""

    # Parse rationale
    rationale_match = re.search(
        r'RATIONALE:\s*(.+)',
        response,
        re.IGNORECASE | re.DOTALL
    )
    rationale = rationale_match.group(1).strip() if rationale_match else response.strip()

    return VerificationResult(label=label, evidence=evidence, rationale=rationale)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Core verify function
# Called by main.py orchestrator, not by claim_extractor.py directly.
# ─────────────────────────────────────────────────────────────────────────────
def verify(fact: str, evidence_chunks: list[str]) -> VerificationResult:
    """
    Verifies a single atomic fact against provided evidence chunks.

    Args:
        fact            : atomic claim string from claim_extractor
        evidence_chunks : list of text chunks from retriever.py

    Returns a VerificationResult. This used to return a bare 3-tuple ordered
    (label, rationale, evidence) — the reverse of what the parser returned —
    which is a swap no type checker or test can see, because all three are
    strings.
    """
    print(f"[Verifier] Verifying: '{fact}'")

    # Empty-evidence short-circuit: with no chunks there is nothing to judge.
    # Asking the model anyway invites a world-knowledge answer — exactly what
    # instruction 4 forbids — so return NOT ENOUGH INFO deterministically.
    if not evidence_chunks:
        print("[Verifier] No evidence chunks — returning NOT ENOUGH INFO without a model call.")
        return VerificationResult(
            label="NOT ENOUGH INFO",
            evidence="",
            rationale="No Wikipedia evidence could be retrieved for this claim.",
        )

    prompt   = build_verification_prompt(fact, evidence_chunks)
    response = call_ollama(prompt)

    print(f"[Verifier] Raw model response:\n{response}")

    # TRUNCATION IS NOT A VERDICT.
    #
    # When generation hits num_predict inside the <think> block, the regex that
    # strips reasoning finds no closing tag, removes everything, and leaves an
    # empty string. parse_verification_response then falls back to its default
    # of NOT ENOUGH INFO — which reads in the final report exactly like a
    # considered judgement that the evidence was inconclusive.
    #
    # Observed on the Jamestown input: 267 seconds of generation, an empty raw
    # response, a blank rationale, and a reported NOT ENOUGH INFO on a claim the
    # model never actually judged. Nothing in the output distinguished it from
    # the two genuine NEI verdicts in the same run.
    #
    # A missing LABEL means the model never answered. Say so, loudly, and put
    # the reason in the rationale so it survives into the report rather than
    # living only in the console.
    if not re.search(r'LABEL:', response, re.IGNORECASE):
        print("[Verifier] NO LABEL IN RESPONSE — generation was truncated or empty. "
              "This is a failure, not a verdict.")
        return VerificationResult(
            label="NOT ENOUGH INFO",
            evidence=evidence_chunks[0] if evidence_chunks else "",
            rationale=(
                "VERIFIER FAILURE: the model produced no LABEL line, most likely "
                "because generation hit the num_predict ceiling inside its "
                "reasoning block. No verdict was reached for this claim — treat "
                "it as unchecked, not as unsupported."
            ),
        )

    result = parse_verification_response(response, evidence_chunks)

    print(f"[Verifier] Label     : {result.label}")
    print(f"[Verifier] Evidence  : {result.evidence}")
    print(f"[Verifier] Rationale : {result.rationale}")

    return result