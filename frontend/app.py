from __future__ import annotations

import streamlit as st

from helpers import fetch_available_symptoms, request_prediction

st.set_page_config(page_title="Medical Recommendation System", page_icon="🩺", layout="wide")

st.title("🩺 Medical Recommendation System")
st.caption("Select symptoms to get an AI-powered disease prediction and healthcare guidance.")


@st.cache_data(ttl=300)
def get_symptom_options() -> list[str]:
    return fetch_available_symptoms()


try:
    symptom_options = get_symptom_options()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Unable to connect to the API. Please run `uvicorn api.main:app --reload` and refresh."
    )
    st.exception(exc)
    st.stop()

selected_symptoms = st.multiselect(
    "Select your symptoms",
    options=symptom_options,
    placeholder="Choose one or more symptoms",
)

predict_clicked = st.button("Predict Disease", type="primary")

if predict_clicked:
    if not selected_symptoms:
        st.warning("Please select at least one symptom before prediction.")
    else:
        with st.spinner("Analyzing symptoms..."):
            try:
                result = request_prediction(selected_symptoms)
            except ValueError as exc:
                st.error("Prediction failed due to invalid input.")
                st.code(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error("Unexpected error while contacting the API.")
                st.exception(exc)
            else:
                st.success("Prediction generated successfully.")

                st.subheader("Predicted Disease")
                st.info(result["predicted_disease"])

                st.subheader("Description")
                st.write(result["description"])

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 💊 Medications")
                    meds = result.get("medications", [])
                    if meds:
                        for item in meds:
                            st.markdown(f"- {item}")
                    else:
                        st.write("No medication guidance available.")

                    st.markdown("### 🥗 Diet Recommendations")
                    diet = result.get("diet", [])
                    if diet:
                        for item in diet:
                            st.markdown(f"- {item}")
                    else:
                        st.write("No diet guidance available.")

                with col2:
                    st.markdown("### ⚠️ Precautions")
                    precautions = result.get("precautions", [])
                    if precautions:
                        for item in precautions:
                            st.markdown(f"- {item}")
                    else:
                        st.write("No precautions available.")

                    st.markdown("### 🏃 Workout Suggestions")
                    workouts = result.get("workout", [])
                    if workouts:
                        for item in workouts:
                            st.markdown(f"- {item}")
                    else:
                        st.write("No workout guidance available.")
