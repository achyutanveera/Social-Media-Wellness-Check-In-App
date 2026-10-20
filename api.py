"""Prediction API for the wellness check-in app.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
Env:  ARTIFACT_DIR (default ./artifacts), ALLOWED_ORIGINS (comma-separated, default *)
"""
import json
import os
from typing import List, Literal

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import preprocess as P

ART = os.environ.get("ARTIFACT_DIR", "artifacts")
meta = json.load(open(f"{ART}/meta.json"))
model_full = joblib.load(f"{ART}/model_full.joblib")
model_nosum = joblib.load(f"{ART}/model_no_sum.joblib")

Likert = Literal["Never", "Rarely", "Sometimes", "Often", "Very Often"]


class CheckIn(BaseModel):
    age: int = Field(ge=5, le=120)
    gender: Literal["Female", "Male", "Other"]
    rs: Literal["Single", "In a relationship", "Married", "Divorced"]
    occupation: Literal["School Student", "University Student", "Salaried Worker", "Retired"]
    sm_type: List[str]
    avg_time_per_day: Literal["Less than 1 hour", "1-2 hours", "2-3 hours",
                              "3-4 hours", "4-5 hours", "More than 5 hours"]
    wo_purpose: Likert
    distracted: Likert
    restless: Likert
    distracted_scale: Likert
    worry_scale: Likert
    concentration: Likert
    compare_to_others: Likert
    compare_feelings: Likert
    validation: Likert
    depressed: Likert
    daily_activity_flux: Likert
    sleeping_issues: Likert


app = FastAPI(title="Wellness check-in scoring")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["POST", "GET"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True, "n_train": meta["n_rows"], "holdout_accuracy": meta["holdout_accuracy"]}


@app.post("/predict")
def predict(c: CheckIn):
    df = P.from_app(c.model_dump())
    impact = int(df[P.ITEMS].sum(axis=1).iloc[0])
    Xf = P.build_features(df, True)[meta["columns_full"]]
    Xn = P.build_features(df, False)[meta["columns_no_sum"]]
    probs = dict(zip(model_nosum.classes_, model_nosum.predict_proba(Xn)[0].round(4).tolist()))
    return {
        "impact_sum": impact,                      # 12-60
        "rule_risk": P.risk_from_impact(impact),   # how the label is defined
        "model_risk_full": str(model_full.predict(Xf)[0]),
        "model_risk_no_sum": str(model_nosum.predict(Xn)[0]),
        "model_probabilities_no_sum": probs,
    }
