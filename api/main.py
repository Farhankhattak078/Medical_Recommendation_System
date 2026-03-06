from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "disease_model.json"
DATA_DIR = BASE_DIR / "data"


class PredictRequest(BaseModel):
    symptoms: list[str] = Field(default_factory=list, min_length=1)


class PredictResponse(BaseModel):
    predicted_disease: str
    description: str
    medications: list[str]
    precautions: list[str]
    diet: list[str]
    workout: list[str]


app = FastAPI(title="Medical Recommendation System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_")


def _split_semicolon(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file was not found at {MODEL_PATH}. "
            "Please ensure /model contains disease_model.json."
        )

    with MODEL_PATH.open("r", encoding="utf-8") as model_file:
        model_data = json.load(model_file)

    symptoms = [_normalize_symptom(symptom) for symptom in model_data["symptoms"]]
    diseases = model_data["diseases"]
    weights = model_data["weights"]
    biases = model_data.get("biases", [0.0] * len(diseases))

    if not (len(diseases) == len(weights) == len(biases)):
        raise RuntimeError("Invalid model shape in disease_model.json")

    symptom_index = {symptom: index for index, symptom in enumerate(symptoms)}

    return {
        "symptoms": symptoms,
        "symptom_index": symptom_index,
        "diseases": diseases,
        "weights": weights,
        "biases": biases,
    }


def _load_recommendations() -> dict[str, dict[str, Any]]:
    recommendations: dict[str, dict[str, Any]] = {}

    with (DATA_DIR / "descriptions.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disease = row["disease"].strip()
            recommendations[disease] = {
                "description": row.get("description", "Description not available.").strip(),
                "medications": [],
                "precautions": [],
                "diet": [],
                "workout": [],
            }

    with (DATA_DIR / "medications.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disease = row["disease"].strip()
            recommendations.setdefault(disease, {})["medications"] = _split_semicolon(
                row.get("medications")
            )

    with (DATA_DIR / "diets.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disease = row["disease"].strip()
            recommendations.setdefault(disease, {})["diet"] = _split_semicolon(
                row.get("diet")
            )

    with (DATA_DIR / "workouts.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disease = row["disease"].strip()
            recommendations.setdefault(disease, {})["workout"] = _split_semicolon(
                row.get("workout")
            )

    with (DATA_DIR / "precautions.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            disease = row["disease"].strip()
            precautions = [
                value.strip()
                for key, value in row.items()
                if key.startswith("precaution") and value and value.strip()
            ]
            recommendations.setdefault(disease, {})["precautions"] = precautions

    return recommendations


def _predict_disease(input_vector: list[int], model_bundle: dict[str, Any]) -> str:
    best_idx = 0
    best_score = float("-inf")

    for disease_idx, disease_weights in enumerate(model_bundle["weights"]):
        score = model_bundle["biases"][disease_idx]
        for symptom_idx, value in enumerate(input_vector):
            score += value * disease_weights[symptom_idx]

        if score > best_score:
            best_score = score
            best_idx = disease_idx

    return model_bundle["diseases"][best_idx]


@app.on_event("startup")
def startup_event() -> None:
    app.state.model_bundle = _load_model()
    app.state.recommendations = _load_recommendations()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Medical Recommendation System API is running.",
        "available_symptoms": app.state.model_bundle["symptoms"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model_bundle = app.state.model_bundle
    normalized_symptoms = [_normalize_symptom(symptom) for symptom in payload.symptoms]

    unknown_symptoms = sorted(
        {symptom for symptom in normalized_symptoms if symptom not in model_bundle["symptom_index"]}
    )
    if unknown_symptoms:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more symptoms are not recognized.",
                "unknown_symptoms": unknown_symptoms,
            },
        )

    input_vector = [0] * len(model_bundle["symptoms"])
    for symptom in set(normalized_symptoms):
        input_vector[model_bundle["symptom_index"][symptom]] = 1

    disease = _predict_disease(input_vector=input_vector, model_bundle=model_bundle)
    recommendation = app.state.recommendations.get(disease, {})

    return PredictResponse(
        predicted_disease=disease,
        description=recommendation.get("description", "Description not available."),
        medications=recommendation.get("medications", []),
        precautions=recommendation.get("precautions", []),
        diet=recommendation.get("diet", []),
        workout=recommendation.get("workout", []),
    )
