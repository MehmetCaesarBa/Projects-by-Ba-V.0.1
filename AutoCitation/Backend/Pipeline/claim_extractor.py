import re
import requests

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
FAST_MODEL   = "phi3:mini"   # swap with your chosen fast model name in Ollama

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 10   # safety ceiling so the loop never runs forever

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
        "options": {"num_ctx": 4096},  # explicit context window to prevent 500 on long iterative prompts
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Parse numbered list from model output
# ─────────────────────────────────────────────────────────────────────────────
def parse_fact_from_response(response: str) -> str | None:
    """
    Extract a single atomic fact from model output.
    Expects a line like:  'Fact: <fact text>'  or just a plain sentence.
    Returns None if the model signals termination.
    """
    # If the model says the claim is fully covered → stop
    if "terminate" in response.lower():
        print("[Extractor] Termination signaled. Loop closed successfully.")
        return None

    # Try to find a line starting with 'Fact:' (as shown in paper Figure 3)
    match = re.search(r'Fact[_\-:]?\s*\d*[:\-]?\s*(.+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: return the first non-empty line
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines[0] if lines else None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build extraction prompt  (mirrors Figure 3 in the AFEV paper)
# ─────────────────────────────────────────────────────────────────────────────
def build_extraction_prompt(
    original_text: str,
    extracted_facts: list[str],
    verification_results: list[str],
    verification_rationales: list[str]  # Crucial addition from Eq. 2
) -> str:
    """
    Dynamic iterative extraction prompt that mirrors Section 4.1 & 5.7 of AFEV.
    Forces the LLM to use prior rationales for entity resolution and context shifting.
    """
    
    # 1. Base Strategy: First Iteration
    if not extracted_facts:
        return f"""You are an expert Atomic Fact Extractor. Your task is to extract the VERY FIRST single, verifiable factual unit from the text.

Rules:
- Isolate one checkable fact containing an active relationship (Entity -> Relation -> Entity).
- Keep all original pronouns or vague descriptors intact for now if they cannot be resolved solely from the text.
- Output exactly in this format: Fact_1: <extracted fact>

Text:
\"\"\"{original_text}\"\"\"

Extract Fact_1:"""

    # 2. Dynamic Strategy: Subsequent Iterations (The Feedback Loop)
    history_block = ""
    for i, (fact, label, rationale) in enumerate(zip(extracted_facts, verification_results, verification_rationales), 1):
        history_block += f"Fact_{i}: {fact}\n"
        history_block += f"Answer_{i}: {label}\n"
        history_block += f"Rationale_{i}: {rationale}\n\n"

    next_idx = len(extracted_facts) + 1

    prompt = f"""You are an adaptive, iterative Atomic Fact Extractor (AFEV Framework). 

Original Text to verify:
\"\"\"{original_text}\"\"\"

Execution History (Decomposed & Verified Sub-layers so far):
{history_block.strip()}

Instructions for Step-by-Step Contextual Extraction:
1. Analyze whether the 'Facts' listed above completely cover the full semantic meaning and all multi-hop dependencies of the Original Text.
2. If the text is fully accounted for, output exactly: Terminate
3. If uncovered details remain, extract the next atomic fact (Fact_{next_idx}).
4. CRITICAL COREFERENCE RESOLUTION: Read the 'Rationale' lines in the Execution History. Use any newly uncovered explicit data (like specific entity names, true identities, or dates) to resolve ambiguous pronouns or relative clauses (e.g., change "that club" to the actual club name mentioned in a prior rationale).

Extract Fact_{next_idx} or Terminate:"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Iterative extractor  (core logic from AFEV Section 4.1)
# ─────────────────────────────────────────────────────────────────────────────
def lightweight_verify(fact: str) -> tuple[str, str]:
    """
    Temporary inline verifier using the fast model.
    Provides real dynamic feedback to the extraction loop
    until verifier.py is integrated.
    
    INTEGRATION POINT: replace this call with verifier.verify(fact)
    Expected signature: verify(fact: str) -> tuple[str, str]  (label, rationale)
    """
    prompt = f"""You are a fact verifier. Assess the following claim based on your general knowledge.

Claim: \"{fact}\"

Rules:
- Output label on the first line: SUPPORTS, REFUTES, or NOT ENOUGH INFO
- Output a one-sentence rationale on the second line explaining your reasoning.
- No extra text, no preamble.

Output format:
LABEL: <SUPPORTS|REFUTES|NOT ENOUGH INFO>
Rationale: <one sentence>"""

    response = call_ollama(prompt)

    # Parse label
    label_match = re.search(r'LABEL:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)', response, re.IGNORECASE)
    label = label_match.group(1).upper() if label_match else "NOT ENOUGH INFO"

    # Parse rationale
    rationale_match = re.search(r'Rationale:\s*(.+)', response, re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else response.strip()

    return label, rationale

# Updated extraction loop setup
def extract_atomic_facts(text: str) -> list[str]:
    extracted_facts: list[str] = []
    verification_results: list[str] = []
    verification_rationales: list[str] = []  # track the rationales sequentially

    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Extractor] Iteration {iteration + 1}")

        # Build prompt using the full historical feedback loop
        prompt = build_extraction_prompt(
            text, 
            extracted_facts, 
            verification_results, 
            verification_rationales
        )
        
        response = call_ollama(prompt)
        print(f"[Extractor] Model response: {response}")

        fact = parse_fact_from_response(response)

        if fact is None:
            print("[Extractor] Termination signal received. Stopping extraction loop.")
            break

        # Grounding sanity check
        if not (set(fact.lower().split()) & set(text.lower().split())):
            # If entity resolution took place, word overlap checking might fail a naive split check. 
            # Better to log a warning rather than blindly skip if you expect text changes.
            print(f"[Extractor] Notice: Structural evolution detected in: '{fact}'")

        extracted_facts.append(fact)
        
        # --- THE CORE SHIFT ---
        # Instead of pushing "PENDING", execute/simulate a verification evaluation step immediately:
        label, rationale = lightweight_verify(fact)
        
        verification_results.append(label)
        verification_rationales.append(rationale)
        
        print(f"[Extractor] Accepted Fact {len(extracted_facts)}: {fact}")
        print(f"[Verifier Mock] Feedback Provided -> Answer: {label} | Rationale: {rationale}")

    return extracted_facts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(text: str) -> list[str]:
    """
    Full claim extraction pipeline:
    precondition → iterative atomic fact extraction → return fact list
    """
    clean_text = precondition(text)
    if clean_text is None:
        return []

    facts = extract_atomic_facts(clean_text)

    print(f"\n[Extractor] Done. {len(facts)} atomic facts extracted:")
    for i, f in enumerate(facts, 1):
        print(f"  {i}. {f}")

    return facts


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(text_ba)