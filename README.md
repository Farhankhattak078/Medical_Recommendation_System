# Medical Recommendation System

A machine-learning powered healthcare recommendation project with:
- **FastAPI backend** for disease prediction and recommendation APIs
- **Streamlit frontend** for an interactive user interface

## Project Structure

```text
Medical_Recommendation_System/
├── api/
│   └── main.py
├── frontend/
│   ├── app.py
│   └── helpers.py
├── data/
│   ├── descriptions.csv
│   ├── medications.csv
│   ├── precautions.csv
│   ├── diets.csv
│   └── workouts.csv
├── model/
│   └── disease_model.json
├── requirements.txt
└── README.md
```

## How It Works

1. Loads a trained disease model from `/model/disease_model.json`.
2. Accepts symptom list input.
3. Converts symptoms to a binary input vector.
4. Predicts the disease from model weights.
5. Fetches disease-specific recommendations from CSV files in `/data`:
   - Description
   - Medications
   - Precautions
   - Diet
   - Workout

## API Endpoints

### `GET /`
Returns health status and available symptom vocabulary.

### `POST /predict`
Request body:

```json
{
  "symptoms": ["fever", "cough", "fatigue"]
}
```

Response body:

```json
{
  "predicted_disease": "Flu",
  "description": "An influenza infection that often causes fever, chills, body ache, and fatigue.",
  "medications": ["Oseltamivir", "Paracetamol", "Rest and hydration"],
  "precautions": ["Stay hydrated", "Take proper rest", "Isolate to prevent spread", "Consult doctor if high fever persists"],
  "diet": ["Light soups", "Citrus fruits", "Electrolyte fluids"],
  "workout": ["Gentle mobility drills after recovery", "Slow walk", "Deep breathing"]
}
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run Backend

```bash
uvicorn api.main:app --reload
```

Backend runs on `http://127.0.0.1:8000`.

## Run Frontend

```bash
streamlit run frontend/app.py
```

Frontend runs on `http://localhost:8501` and uses backend API at `http://127.0.0.1:8000` by default.

To use a custom backend URL:

```bash
API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py
```

## Error Handling Included

- Empty symptom selection warning in UI.
- Invalid symptom handling from API (`422` response).
- API connectivity error handling in Streamlit.

