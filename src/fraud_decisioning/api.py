from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Decisioning API", version="1.4.0")

class RiskRequest(BaseModel):
    risk_probability: float
    amount: float
    review_threshold: float = 0.20
    block_threshold: float = 0.70

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/decision")
def decision(req: RiskRequest):
    p = min(max(req.risk_probability, 0.0), 1.0)
    if p >= req.block_threshold:
        action = "block"
    elif p >= req.review_threshold:
        action = "review"
    else:
        action = "approve"
    return {"risk_probability": p, "amount": req.amount, "action": action}
