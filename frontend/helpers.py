from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def fetch_available_symptoms() -> list[str]:
    response = requests.get(f"{API_URL}/", timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload.get("available_symptoms", [])


def request_prediction(selected_symptoms: list[str]) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/predict",
        json={"symptoms": selected_symptoms},
        timeout=20,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(detail)
    return response.json()
