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
# budget must also cover the <think> block, hence the conditional.
VERIFIER_NUM_PREDICT = 256 if VERIFIER_THINKING is False else 1024


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
6. Only output REFUTES when a chunk explicitly and unambiguously contradicts the claim.

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

    result = parse_verification_response(response, evidence_chunks)

    print(f"[Verifier] Label     : {result.label}")
    print(f"[Verifier] Evidence  : {result.evidence}")
    print(f"[Verifier] Rationale : {result.rationale}")

    return result