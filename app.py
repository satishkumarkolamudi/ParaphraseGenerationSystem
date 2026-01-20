from fastapi import FastAPI
from pydantic import BaseModel, Field
from cpg import custom_paraphrase
from llm_paraphraser import llm_paraphrase
from metrics import (
    measure_latency,
    semantic_similarity,
    bleu,
    readability,
)


app = FastAPI(
    title="Paraphrase Service",
    version="0.1.0",
    description=(
        "Service that produces paraphrases using a heuristic CPG module and an LLM wrapper, "
        "and returns simple quality metrics (semantic similarity, BLEU, readability) and latencies."
    ),
)


class ParaphraseRequest(BaseModel):
    text: str = Field(..., examples=["The quick brown fox jumps over the lazy dog."])


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    message: str = Field(..., examples=["Paraphrase service is running"])


class ParaphraseResult(BaseModel):
    output: str = Field(..., examples=["A paraphrased version of the input"])
    words: int = Field(..., examples=[9])
    semantic_similarity: float = Field(..., examples=[0.85])
    bleu: float = Field(..., examples=[0.45])
    readability: float = Field(..., examples=[72.3])
    latency_ms: float = Field(..., examples=[123.4567])


class ParaphraseResponse(BaseModel):
    input_words: int = Field(..., examples=[9])
    cpg: ParaphraseResult
    llm: ParaphraseResult


@app.get(
    "/",
    tags=["health"],
    summary="Health check",
    response_model=HealthResponse,
    operation_id="health_check",
    description="Return service health status and a short message.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "message": "Paraphrase service is running"}
                }
            },
        }
    },
)
async def root():
    return {"status": "ok", "message": "Paraphrase service is running"}


@app.post(
    "/paraphrase",
    tags=["paraphrase"],
    summary="Paraphrase text with CPG and LLM",
    description=(
        "Run the heuristic CPG paraphraser and the LLM paraphraser on the provided text. "
        "Returns paraphrased outputs plus metrics (semantic similarity, BLEU, readability) and per-model latency in ms."
    ),
    response_model=ParaphraseResponse,
    operation_id="paraphrase_text",
    responses={
        200: {
            "description": "Paraphrase results for both CPG and LLM",
            "content": {
                "application/json": {
                    "example": {
                        "input_words": 9,
                        "cpg": {
                            "output": "A quick brown fox leaped over the lazy dog.",
                            "words": 9,
                            "semantic_similarity": 0.82,
                            "bleu": 0.47,
                            "readability": 72.1,
                            "latency_ms": 45.1234,
                        },
                        "llm": {
                            "output": "The swift brown fox jumped over the idle dog.",
                            "words": 9,
                            "semantic_similarity": 0.88,
                            "bleu": 0.51,
                            "readability": 69.5,
                            "latency_ms": 2045.9876,
                        },
                    }
                }
            },
        },
        400: {"description": "Bad request - invalid input"},
        500: {"description": "Server error"},
    },
)
async def paraphrase(req: ParaphraseRequest):
    # req = req.model_dump()
    # print(req)
    text = req.text or ""

    result = {
        "input_words": len(text.split()),
        "cpg": {},
        "llm": {},
    }

    # CPG
    cpg_output, cpg_time = measure_latency(custom_paraphrase, text)

    # LLM
    llm_output, llm_time = measure_latency(llm_paraphrase, text)

    # Ensure outputs are strings (safety in case of None)
    cpg_output = cpg_output or ""
    llm_output = llm_output or ""

    result["cpg"] = {
        "output": str(cpg_output),
        "words": len(str(cpg_output).split()),
        "semantic_similarity": float(semantic_similarity(text, str(cpg_output))),
        "bleu": float(bleu(text, str(cpg_output))),
        "readability": float(readability(str(cpg_output))),
        "latency_ms": round(float(cpg_time), 4),
    }

    result["llm"] = {
        "output": str(llm_output),
        "words": len(str(llm_output).split()),
        "semantic_similarity": float(semantic_similarity(text, str(llm_output))),
        "bleu": float(bleu(text, str(llm_output))),
        "readability": float(readability(str(llm_output))),
        "latency_ms": round(float(llm_time), 4),
    }

    return result
