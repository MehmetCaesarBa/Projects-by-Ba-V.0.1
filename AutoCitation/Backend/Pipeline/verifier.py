import re
from dataclasses import dataclass

import requests

import models.ollama_client as ollama_client

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
# before its answer, which call_ollama() then strips and discards.
#
# MEASURED, not assumed. Ollama's own counters on a real run:
#
#   [Inference] qwen3:8b  output 312 tok in 64.5s (4.8 tok/s)
#   [Inference] qwen3:8b  output 303 tok in 62.2s (4.9 tok/s)
#
# A LABEL / EVIDENCE / RATIONALE answer is ~55 tokens. So ~250 tokens per call
# — 80% of the output — were reasoning, generated at 4.8 tok/s on CPU (~52s
# each) and then deleted. One two-claim request spent ~104s producing text that
# never left the function.
#
# REVERTED to "low" after False caused a measured quality regression.
#
# With False, an input that previously produced SUPPORTS + REFUTES + NEI came
# back as three SUPPORTS. The lost verdict was:
#
#   claim:    "Jamestown ... the earliest European permanent settlement in
#              what is now the United States"
#   evidence: "The Spanish were the first Europeans to establish a permanent
#              settlement in what became the United States, at Saint Augustine,
#              Florida (1565)."
#   was: REFUTES     became: SUPPORTS
#
# Refuting that requires chaining three steps — Spanish are European, 1565
# precedes 1607, therefore "earliest" is false. Nothing in the evidence says
# "Jamestown was not first". Reasoning tokens are what buy that chain, and
# without them qwen3 sees topically-related evidence and agrees.
#
# The speed was real (312 output tokens -> 44, decode 64.5s -> 8.9s) but it was
# bought with the system's ability to detect contradiction, which is the whole
# point of the system. Speed that costs REFUTES is not speed worth having.
VERIFIER_THINKING = "low"

# Evidence chunks are long Wikipedia passages; the previous call sent no
# num_ctx at all and silently inherited Ollama's default window, which risks
# truncating the very evidence the verdict depends on.
VERIFIER_NUM_CTX = 8192

# How long Ollama keeps the model in memory after a call. The default is 5
# minutes; a 5 GB reload costs ~34s on this hardware.
KEEP_ALIVE = "30m"

# Output ceiling. With thinking disabled the answer is three short lines, so a
# tight cap costs nothing. With thinking on the budget must ALSO cover the
# <think> block — and 1024 turned out to be too tight:
#
#   [Inference] qwen3:8b  output 1024 tok in 263.7s
#   [Verifier] Raw model response:        <- empty
#
# Exactly at the ceiling, cut off mid-reasoning, 263 seconds for nothing. The
# parser then defaulted to NOT ENOUGH INFO, so a truncated generation was
# indistinguishable from a considered verdict. Raised, and verify() now detects
# the case explicitly rather than letting it masquerade as a label.
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
        # Keep the model resident between calls. A measured run showed
        # `load 34.4s` on the first qwen3 call and 0.0s afterwards — a 5 GB
        # read from disk. Ollama's default keep_alive is 5 minutes, so an
        # interactive user who pauses between requests pays that again every
        # time. Requesting it per-call avoids depending on a server-wide
        # environment variable being set.
        "keep_alive": KEEP_ALIVE,
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
    body = response.json()
    ollama_client.log_inference_stats(body, REASONING_MODEL)
    raw = body["response"]

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

    RULE 6 IS THE OVER-REFUTATION FIX. Observed:

        claim    "The title of the world's longest river belongs to the Amazon."
        evidence "...the second-longest or longest river system in the world,
                  a title which is disputed with the Nile."
        verdict  REFUTES
        reason   "...directly contradicting the claim that the Amazon holds the
                  title UNAMBIGUOUSLY."

    'Unambiguously' is not in the claim. The model supplied a strength qualifier
    the claim never asserted and then refuted its own addition. That is the
    whole mechanism, and it explains why the previous wording — "only output
    REFUTES when a chunk explicitly and unambiguously contradicts the claim" —
    did not prevent it: the model believed it HAD found an unambiguous
    contradiction, of a claim it had silently strengthened.

    So the rule now constrains two things instead of one: what counts as a
    contradiction (something that cannot hold simultaneously, not something that
    declines to confirm), and what the claim is allowed to mean (exactly what it
    says). Evidence that reports a question as OPEN is named explicitly, because
    'disputed' is the textbook NOT ENOUGH INFO signal and nothing in the prompt
    had ever said so.

    Kept inside rule 6 rather than added as rules 7 and 8 on purpose. The
    withdrawn rule 7 (below) failed partly because seven unordered instructions
    were competing for attention; piling more on would repeat that.

    SIX RULES, NOT SEVEN — a rule 7 was added and then withdrawn. The history
    is kept here because the withdrawal is provisional, and re-adding it without
    reading this would repeat the mistake.

    Rule 7 ("CHECK THE QUALIFIERS") was written against a real failure. The
    verifier grants SUPPORTS on evidence for a NARROWER statement than the claim
    makes; observed twice:

        claim "earliest EUROPEAN settlement"  <- evidence "first ENGLISH settlement"
        claim "highest FROM BASE TO PEAK"     <- evidence "highest ABOVE SEA LEVEL"

    In the first case the model's own rationale read "...the first permanent
    ENGLISH settlement in the Americas, which includes the United States,
    directly supporting the claim" — it wrote the mismatched word itself,
    checked the geographic scope correctly, and never checked the nationality
    scope at all. That is a PARTIAL comparison: it verifies whichever dimensions
    it happens to notice and stops.

    WHY IT WAS WITHDRAWN. In the next run the verifier over-corrected in the
    opposite direction. Given

        claim    "The title of the world's longest river belongs to the Amazon."
        evidence "...the second-longest or longest river system in the world,
                  a title which is disputed with the Nile."

    it returned REFUTES, with the rationale "...directly contradicting the claim
    that the Amazon holds the title unambiguously." The word 'unambiguously' is
    NOT IN THE CLAIM. The model invented a strength qualifier and refuted its own
    invention — which is exactly what a rule that says "compare the qualifying
    terms word by word" invites when the claim has no qualifier to compare. The
    same input had previously returned NOT ENOUGH INFO, which is correct.

    Rule 7 closed with "the answer is NOT ENOUGH INFO", so it did not ask for
    this; rule 6 ("only REFUTES when explicit and unambiguous") did not prevent
    it either. Seven unordered instructions were competing, and the outcome was
    no longer predictable from any one of them.

    So it is removed to ISOLATE THE VARIABLE, not because the problem it
    addressed is solved — the partial-comparison failure is still live and still
    unhandled. Re-running the disputed-river input against these six rules says
    whether rule 7 caused the REFUTES. If it did, the replacement needs to be
    narrower: a rule about EXTRA qualifiers in the evidence, not a general
    instruction to hunt for qualifier mismatches, plus an explicit clause that
    evidence describing a question as open or disputed is NOT ENOUGH INFO rather
    than a contradiction.

    Any replacement's examples must come from domains the evaluation inputs
    never touch. The withdrawn version used French novels, Paris towers and
    African lakes for that reason; a test-set sentence in the prompt shows the
    model the answer to a question it is about to be asked, and that case then
    measures nothing.

    CLAIM FIRST, DELIBERATELY — do not reorder this for prefix caching.

    It was reordered once, putting the constant instructions first so Ollama's
    KV cache could serve them. That is the right move for phi3, which gained
    17x on prefill. On qwen3 it backfired badly, because qwen3 REASONS, and
    where the claim sits changes how long it reasons:

        claim first (this version)   output 312, 303 tok   decode  ~64s
        instructions first           output 1024, 776 tok  decode ~264s

    The 1024 is not a coincidence — it is VERIFIER_NUM_PREDICT. Reading all the
    instructions before seeing what it was judging made the model deliberate
    until it hit the ceiling, got truncated mid-<think>, and returned an EMPTY
    response that the parser then defaulted to NOT ENOUGH INFO. A correct
    SUPPORTS became a silent failure after 263 seconds of wasted generation.

    The lesson generalises: prompt ordering is safe to optimise for a model
    that only completes, and is not safe for a model that reasons, because
    ordering changes the reasoning budget it decides to spend.
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

Answer immediately. Do not deliberate at length — the decision is a direct comparison of the claim against the chunks.

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
    # strips reasoning leaves an empty string, and parse_verification_response
    # falls back to NOT ENOUGH INFO — which reads in the final report exactly
    # like a considered judgement that the evidence was inconclusive. One
    # observed run spent 263 seconds producing nothing and reported NEI against
    # a chunk that plainly said "founded on May 14, 1607".
    #
    # A missing LABEL means the model never answered. Say so.
    if not re.search(r'LABEL:', response, re.IGNORECASE):
        print("[Verifier] NO LABEL IN RESPONSE — generation was truncated or empty. "
              "This is a failure, not a verdict.")
        return VerificationResult(
            label="NOT ENOUGH INFO",
            evidence=evidence_chunks[0] if evidence_chunks else "",
            rationale=(
                "VERIFIER FAILURE: the model produced no LABEL line, most likely "
                "because generation hit the num_predict ceiling inside its "
                "reasoning block. No verdict was reached for this claim."
            ),
        )

    result = parse_verification_response(response, evidence_chunks)

    print(f"[Verifier] Label     : {result.label}")
    print(f"[Verifier] Evidence  : {result.evidence}")
    print(f"[Verifier] Rationale : {result.rationale}")

    return result