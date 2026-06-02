import re
import requests

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
REASONING_MODEL = "qwen3:8b"

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Call Ollama (reasoning model)
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """Send a prompt to the reasoning model and return the raw response string."""
    payload = {
        "model": REASONING_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()


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

Output strictly in this format with no extra text:
LABEL: <SUPPORTS|REFUTES|NOT ENOUGH INFO>
EVIDENCE: <copy the single most relevant evidence chunk exactly>
RATIONALE: <one sentence explaining why the label was assigned based on the evidence>"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Parse model output
# ─────────────────────────────────────────────────────────────────────────────
def parse_verification_response(
    response: str,
    evidence_chunks: list[str]
) -> tuple[str, str, str]:
    """
    Parses the three structured fields from the reasoning model response.
    Falls back gracefully if any field is missing or malformed.

    Returns:
        label     : SUPPORTS | REFUTES | NOT ENOUGH INFO
        evidence  : the chunk the model selected
        rationale : one sentence explanation
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

    # Parse evidence chunk
    evidence_match = re.search(
        r'EVIDENCE:\s*(.+?)(?=\nRATIONALE:|\Z)',
        response,
        re.IGNORECASE | re.DOTALL
    )
    evidence = evidence_match.group(1).strip() if evidence_match else evidence_chunks[0]

    # Parse rationale
    rationale_match = re.search(
        r'RATIONALE:\s*(.+)',
        response,
        re.IGNORECASE | re.DOTALL
    )
    rationale = rationale_match.group(1).strip() if rationale_match else response.strip()

    return label, evidence, rationale


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Core verify function
# Called by main.py orchestrator, not by claim_extractor.py directly.
# ─────────────────────────────────────────────────────────────────────────────
def verify(fact: str, evidence_chunks: list[str]) -> tuple[str, str, str]:
    """
    Verifies a single atomic fact against provided evidence chunks.

    Args:
        fact            : atomic claim string from claim_extractor
        evidence_chunks : list of text chunks from retriever.py

    Returns:
        label     : SUPPORTS | REFUTES | NOT ENOUGH INFO
        rationale : one sentence explanation
        evidence  : the chunk that drove the decision
    """
    print(f"[Verifier] Verifying: '{fact}'")

    prompt   = build_verification_prompt(fact, evidence_chunks)
    response = call_ollama(prompt)

    print(f"[Verifier] Raw model response:\n{response}")

    label, evidence, rationale = parse_verification_response(response, evidence_chunks)

    print(f"[Verifier] Label     : {label}")
    print(f"[Verifier] Evidence  : {evidence}")
    print(f"[Verifier] Rationale : {rationale}")

    return label, rationale, evidence