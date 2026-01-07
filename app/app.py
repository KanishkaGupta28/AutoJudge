import streamlit as st
import joblib
import os
import numpy as np

# ------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------
st.set_page_config(
    page_title="AutoJudge",
    layout="centered"
)

st.title(" AutoJudge: Programming Problem Difficulty Predictor")
st.write("Paste the programming problem details below to get predictions.")

# ------------------------------------------------
# Correct Paths
# ------------------------------------------------
# Current file = AutoJudge/app/app.py
# Models       = AutoJudge/models/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ------------------------------------------------
# Load Models
# ------------------------------------------------
@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    clf = joblib.load(os.path.join(MODELS_DIR, "difficulty_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "difficulty_regressor.pkl"))
    return vectorizer, clf, reg

vectorizer, clf, reg = load_models()

# ------------------------------------------------
# Input Section
# ------------------------------------------------
description = st.text_area("Problem Description")
input_desc = st.text_area(" Input Description")
output_desc = st.text_area(" Output Description")

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.button("Predict Difficulty"):
    combined_text = f"{description} {input_desc} {output_desc}".strip()

    if not combined_text:
        st.warning("⚠️ Please fill all text fields before predicting.")
    else:
        # TF-IDF Transform
        X = vectorizer.transform([combined_text])

        # ---- Classification ----
        pred_class = clf.predict(X)[0]
        final_class = pred_class.capitalize()

        # ---- Regression ----
        pred_score = float(reg.predict(X)[0])

        # ---- Confidence ----
        confidence = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            confidence = np.max(probs) * 100

        # ------------------------------------------------
        # 🔥 Consistency Fix (Score + Class Agreement)
        # ------------------------------------------------
        if pred_score <= 4.5:
            score_class = "Easy"
        elif pred_score <= 6.5:
            score_class = "Medium"
        else:
            score_class = "Hard"

        # If classifier is unsure → trust score
        if confidence is not None and confidence < 50:
            final_class = score_class
        else:
            # Prevent contradictions like Hard + 5.2
            if final_class == "Hard" and pred_score < 6.5:
                final_class = "Medium"
            if final_class == "Easy" and pred_score > 4.8:
                final_class = "Medium"

        # ------------------------------------------------
        # Output
        # ------------------------------------------------
        st.success(f" Predicted Difficulty Class: **{final_class}**")
        st.info(f" Predicted Difficulty Score: **{pred_score:.2f}**")

        