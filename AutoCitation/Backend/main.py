from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import models.ollama_client as ollama_client
import Pipeline.claim_extractor as claim_extractor
import Pipeline.ner as ner
import Pipeline.retriever as retriever
import Pipeline.verifier as verifier
import Pipeline.postprocessor as postprocessor
from config import MAX_INPUT_WORDS


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────
class CheckRequest(BaseModel):
    """
    Incoming request body for the POST /check endpoint.

    Fields:
        text : the user's input paragraph to fact-check.
                Validated here rather than inside claim_extractor so that
                FastAPI returns a structured 422 Unprocessable Entity with
                a clear message before any pipeline work begins.

    Validation:
        - Must be a non-empty string after stripping whitespace.
        - Word count must not exceed MAX_INPUT_WORDS (from config.py).
          Word count is checked here as a fast pre-flight guard; the
          deeper character-level validation (allowed characters, Turkish
          Unicode) is still performed inside claim_extractor.precondition()
          so that module remains independently testable.
    """
    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Input text must not be empty.")

        word_count = len(v.split())
        if word_count > MAX_INPUT_WORDS:
            raise ValueError(
                f"Input exceeds the {MAX_INPUT_WORDS}-word limit "
                f"({word_count} words received). "
                f"Please shorten your text and try again."
            )
        return v


class FactResult(BaseModel):
    """
    Single per-fact verification result returned inside CheckResponse.

    All fields are populated by postprocessor.assemble_result() and passed
    through here unchanged. Defining the shape explicitly as a Pydantic model
    means FastAPI validates the postprocessor's output before serialization
    and generates accurate OpenAPI schema docs automatically.

    Fields:
        claim       : the atomic fact string extracted from the input
        label       : SUPPORTS | REFUTES | NOT ENOUGH INFO
        label_color : CSS class token for frontend color-coding (green/red/yellow)
        rationale   : one-sentence explanation from the reasoning model
        evidence    : truncated evidence chunk that drove the decision
        source_url  : canonical Wikipedia article URL (or fallback search URL)
    """
    claim       : str
    label       : str
    label_color : str
    rationale   : str
    evidence    : str
    source_url  : str


class SummaryBlock(BaseModel):
    """Label frequency counts across all facts in a single /check response."""
    SUPPORTS        : int
    REFUTES         : int
    NOT_ENOUGH_INFO : int = 0

    class Config:
        # Allow the postprocessor's "NOT ENOUGH INFO" key (with spaces)
        # to map to the Python-safe "NOT_ENOUGH_INFO" field name.
        populate_by_name = True


class CheckResponse(BaseModel):
    """
    Top-level response body for POST /check.

    Mirrors the aggregated output shape produced by postprocessor.aggregate()
    with the addition of Pydantic field validation for safe serialization.

    Fields:
        total_claims  : number of atomic facts processed
        overall_label : single top-level verdict for the full input text
        summary       : per-label counts across all facts
        results       : ordered list of per-fact FactResult objects
    """
    total_claims  : int
    overall_label : str
    summary       : dict[str, int]
    results       : list[FactResult]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Application Lifespan (Startup Checks)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs startup logic before the server
    begins accepting requests, and teardown logic on shutdown.

    Startup checks (fail-fast philosophy):
    Both checks are informational rather than blocking — the server starts
    regardless of their outcome so that a developer can still hit /check
    and see a clear error message rather than a refused connection.
    Blocking startup on model availability would prevent the server from
    running even when Ollama is temporarily restarting mid-development.

    1. ollama_client.health_check()
       Pings localhost:11434 to confirm Ollama is running. Prints a
       "run: ollama serve" hint if unreachable.

    2. ollama_client.check_models()
       Queries /api/tags to confirm both phi3:mini and qwen3:8b have been
       pulled. Prints "run: ollama pull <model>" per missing model.

    These checks surface the two most common local setup mistakes (Ollama
    not started, models not pulled) in the server log at startup rather
    than as cryptic ConnectionRefused errors during the first request.
    """
    print("\n[Main] AutoCitation pipeline starting up...")

    ollama_client.health_check()
    ollama_client.check_models()

    print("[Main] Startup checks complete. Server ready.\n")
    yield
    print("[Main] AutoCitation pipeline shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AutoCitation API",
    description = (
        "Fact-checks a text paragraph by extracting atomic claims, "
        "retrieving Wikipedia evidence, and classifying each claim as "
        "SUPPORTS, REFUTES, or NOT ENOUGH INFO using local Ollama models."
    ),
    version  = "0.1.0",
    lifespan = lifespan,
)

# CORS middleware: permits the plain HTML/JS frontend (index.html opened
# directly from the file system as a file:// URL, or served on a different
# port during development) to call the FastAPI backend without browser
# cross-origin blocking. For PoC, all origins are allowed. In production
# this should be tightened to the specific frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["POST", "GET"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Per-Fact Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_fact_pipeline(fact: str) -> dict:
    """
    Executes the full retrieval-verification pipeline for a single atomic fact.

    Called once per fact inside the /check endpoint's fact loop. Isolating
    per-fact logic here keeps the endpoint handler readable and makes each
    stage independently traceable in logs.

    Pipeline stages for one fact:
        1. ner.extract_query()      → search string for Wikipedia + postprocessor URL
        2. retriever.fetch()        → top-k evidence chunks (may be empty)
        3. verifier.verify()        → label, rationale, evidence chunk

    Wikipedia-miss handling (empty chunks):
    If retriever.fetch() returns an empty list — because Wikipedia found no
    relevant articles for the NER query — verifier.verify() is still called
    with an empty evidence list. The verifier's prompt instructs it to output
    NOT ENOUGH INFO when no evidence is available, which is the correct and
    honest response: the system cannot support or refute a claim it found no
    source for. Skipping the fact entirely would silently reduce the result
    count and misrepresent how many claims were actually checked.

    NER double-call note:
    ner.extract_query() is called here explicitly (in addition to the
    internal call inside retriever.fetch()) so that main.py can capture
    the query string and pass it to postprocessor.assemble_result() for
    Wikipedia URL resolution. The NER step is pure CPU/spaCy with no model
    inference cost, so the duplicate call adds negligible latency at PoC scale.
    A future refactor of retriever.fetch() to return (chunks, query) as a
    tuple would eliminate this duplication.

    Args:
        fact : atomic claim string from claim_extractor.run()

    Returns:
        Raw pipeline output dict with keys:
            claim, label, rationale, evidence, ner_query
        Ready to be passed into postprocessor.process().

    Raises:
        Does not raise — any exception from retriever or verifier is caught
        and converted into a NOT ENOUGH INFO result so one failing fact does
        not abort the entire /check response.
    """
    print(f"\n[Main] Running pipeline for fact: '{fact}'")

    # Stage 1 — NER query (captured here for postprocessor URL resolution)
    ner_query = ner.extract_query(fact)

    # Stage 2 — Wikipedia retrieval (empty list on miss; handled in Stage 3)
    try:
        chunks = retriever.fetch(fact)
    except Exception as e:
        print(f"[Main] Retriever failed for fact '{fact}': {e}")
        chunks = []

    if not chunks:
        print(
            f"[Main] No evidence chunks retrieved for fact '{fact}'. "
            f"Proceeding to verifier — will yield NOT ENOUGH INFO."
        )

    # Stage 3 — Verification (called regardless of chunk availability)
    try:
        label, rationale, evidence = verifier.verify(fact, chunks)
    except Exception as e:
        print(f"[Main] Verifier failed for fact '{fact}': {e}")
        label     = "NOT ENOUGH INFO"
        rationale = "Verification could not be completed due to an internal error."
        evidence  = ""

    return {
        "claim"     : fact,
        "label"     : label,
        "rationale" : rationale,
        "evidence"  : evidence,
        "ner_query" : ner_query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — POST /check Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/check", response_model=CheckResponse)
def check(request: CheckRequest) -> CheckResponse:
    """
    POST /check — Full AutoCitation pipeline endpoint.

    Accepts a paragraph of text, extracts atomic claims, retrieves Wikipedia
    evidence per claim, verifies each claim, and returns a structured
    verdict for the full input.

    Request body (JSON):
        { "text": "<paragraph up to 500 words>" }

    Response body (JSON):
        {
            "total_claims"  : int,
            "overall_label" : "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
            "summary"       : {"SUPPORTS": int, "REFUTES": int, "NOT ENOUGH INFO": int},
            "results": [
                {
                    "claim"       : str,
                    "label"       : "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
                    "label_color" : "green" | "red" | "yellow",
                    "rationale"   : str,
                    "evidence"    : str,
                    "source_url"  : str
                },
                ...
            ]
        }

    Error responses:
        422 Unprocessable Entity : input failed Pydantic validation
                                   (empty text, exceeds word limit)
        503 Service Unavailable  : claim extraction returned zero facts
                                   (precondition failure: invalid characters,
                                   or model returned no extractable claims)
        500 Internal Server Error: unexpected exception during pipeline
                                   (surfaced with detail message for debugging)

    Pipeline flow:
        CheckRequest.text
            → claim_extractor.run()          # iterative atomic fact extraction
            → [for each fact]
                → run_fact_pipeline(fact)    # NER + retrieval + verification
            → postprocessor.process()        # assemble + aggregate → response
    """
    print(f"\n[Main] POST /check — received {len(request.text.split())} words.")

    # ── Stage 1: Claim Extraction ─────────────────────────────────────────────
    try:
        facts = claim_extractor.run(request.text)
    except Exception as e:
        print(f"[Main] Claim extraction raised an unexpected exception: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Claim extraction failed: {e}"
        )

    if not facts:
        # precondition() returned None (bad characters / word count) or
        # the extraction loop produced zero facts from the input.
        raise HTTPException(
            status_code = 503,
            detail      = (
                "No verifiable claims could be extracted from the input. "
                "Check that the text is within 500 words and contains "
                "factual statements rather than only opinions or questions."
            )
        )

    print(f"[Main] {len(facts)} fact(s) extracted. Starting per-fact pipeline.")

    # ── Stage 2: Per-Fact Retrieval + Verification ────────────────────────────
    # Facts are processed sequentially. A parallel implementation (asyncio
    # gather or a thread pool) would reduce latency on multi-fact inputs but
    # is deferred until the PoC baseline is validated — concurrent Ollama
    # calls on a single GPU would queue anyway and offer no real speedup
    # until a multi-GPU or batched inference setup is in place.
    pipeline_outputs = []
    for fact in facts:
        result = run_fact_pipeline(fact)
        pipeline_outputs.append(result)

    # ── Stage 3: Post-Processing ──────────────────────────────────────────────
    try:
        final_response = postprocessor.process(pipeline_outputs)
    except Exception as e:
        print(f"[Main] Post-processing failed: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Post-processing failed: {e}"
        )

    print(
        f"[Main] /check complete — "
        f"{final_response['total_claims']} claim(s) | "
        f"overall: {final_response['overall_label']}"
    )

    return final_response