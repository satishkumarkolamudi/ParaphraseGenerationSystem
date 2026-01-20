from fastapi import FastAPI
from pydantic import BaseModel
from cpg import custom_paraphrase
from llm_paraphraser import llm_paraphrase
from metrics import (
    measure_latency,
    semantic_similarity,
    bleu,
    readability,
)

app = FastAPI(title="Paraphrase Service")


class ParaphraseRequest(BaseModel):
    text: str


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "message": "Paraphrase service is running"}


@app.post("/paraphrase", tags=["paraphrase"])
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

    result["cpg"] = {
        "output": cpg_output,
        "words": len(cpg_output.split()),
        "semantic_similarity": float(semantic_similarity(text, cpg_output)),
        "bleu": float(bleu(text, cpg_output)),
        "readability": float(readability(cpg_output)),
        "latency_ms": round(float(cpg_time), 4)
    }

    result["llm"] = {
        "output": llm_output,
        "words": len(llm_output.split()),
        "semantic_similarity": float(semantic_similarity(text, llm_output)),
        "bleu": float(bleu(text, llm_output)),
        "readability": float(readability(llm_output)),
        "latency_ms": round(float(llm_time), 4)
    }

    return result
