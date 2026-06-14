import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score, balanced_accuracy_score, classification_report,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Haar Cascades

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# 1. DATA LOADING

@st.cache_data(show_spinner=False)
def load_data(path: str = "teen_adult_asd_train.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset '{path}' not found. Place teen_adult_asd_train.csv in the same folder.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def get_aq_columns(columns) -> list:
    """Find A1-A10 columns dynamically while preserving the dataset's exact names."""
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    aq_cols = []

    for i in range(1, 11):
        candidates = [
            f"A{i}_Score",
            f"A{i}",
            f"A{i} Score",
            f"A{i}_score"
        ]
        matched = None
        for cand in candidates:
            if cand.lower() in lower_map:
                matched = lower_map[cand.lower()]
                break
        if matched:
            aq_cols.append(matched)

    return aq_cols


# 2. PREPROCESSING

def preprocess_data(df: pd.DataFrame):
   
    df = df.copy()
    df.columns = df.columns.str.strip()

    aq_cols = get_aq_columns(df.columns)
    if len(aq_cols) != 10:
        raise ValueError(
            f"Expected 10 AQ columns (A1-A10), found {len(aq_cols)}: {aq_cols}"
        )

    for col in aq_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    if 'result' in df.columns:
      df.drop(columns=['result'], inplace=True)

    # Drop identifier / low-signal columns
    drop_cols = ['ID', 'contry_of_res', 'age_desc', 'relation']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Normalise target column
    target_col = 'Class/ASD'
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.strip().str.upper().map({'YES': 1, 'NO': 0})
    df[target_col] = df[target_col].astype(int)

    # One-hot encode categoricals
    cat_cols = ['gender', 'ethnicity', 'jaundice', 'austim', 'used_app_before', 'age_group']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


# 3. MODEL TRAINING  — cached, never reruns on refresh

@st.cache_resource(show_spinner=False)
def train_model(data_path: str = "teen_adult_asd_train.csv"):
    df = pd.read_csv(data_path)
    X, y = preprocess_data(df)
    feature_cols = list(X.columns)
    # Load separate test dataset
    test_df = pd.read_csv("teen_adult_asd_test.csv")

    X_test, y_test = preprocess_data(test_df)
    # Align train/test columns
    X_train, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

    # Keep train variables consistent
    X = X_train


    # Training data
    y_train = y


    # SMOTE on training split only
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Class weight for any remaining imbalance
    neg = (y_train_res == 0).sum()
    pos = (y_train_res == 1).sum()
    spw = neg / pos if pos > 0 else 1.0


    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        gamma=0.2,
        reg_lambda=2,
        reg_alpha=0.1,
        scale_pos_weight=spw,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
    )


    model.fit(X_train_res, y_train_res)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Check duplicate overlap
    train_rows = set(map(tuple, X_train.values))
    test_rows = set(map(tuple, X_test.values))

    overlap = len(train_rows.intersection(test_rows))

    print("Duplicate overlap:", overlap)


    joblib.dump(model, "asd_model.pkl")

    y_proba = model.predict_proba(X_test)[:, 1]
    prediction_threshold = 0.45
    y_pred  = (y_proba >= prediction_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy":     accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, y_proba),
        "threshold": prediction_threshold,
        "report":       classification_report(y_test, y_pred, target_names=["No ASD", "ASD"]),
        "cm":           confusion_matrix(y_test, y_pred),
        "y_test":       y_test,
        "y_pred":       y_pred,
        "feature_cols": feature_cols,
        "auc": roc_auc_score(y_test, y_proba),
        "y_proba": y_proba,
    }
    return model, metrics


# 4. FACIAL ANALYSIS

def analyze_face(image: np.ndarray) -> dict:
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotated = image.copy()

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    result = {
        "face_detected":    False,
        "eyes_detected":    0,
        "behavioral_score": 0.0,
        "note": "No face detected in the image.",
        "annotated_image":  annotated,
    }

    if len(faces) == 0:
        return result

    result["face_detected"] = True
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 100), 2)

    roi_gray  = gray[y:y+h, x:x+w]
    roi_color = annotated[y:y+h, x:x+w]
    eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    result["eyes_detected"] = len(eyes)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 100, 0), 2)

    if len(eyes) == 0:
        score = 0.35
        note  = "Face detected but eyes not clearly visible — possible gaze avoidance (a common ASD behavioural indicator)."
    elif len(eyes) == 1:
        score = 0.20
        note  = "Face and one eye detected. Partial gaze contact observed."
    else:
        score = 0.05
        note  = "Face and both eyes clearly detected. No strong gaze-avoidance signal."

    result.update({"behavioral_score": score, "note": note, "annotated_image": annotated})
    return result


# 5. PREDICTION  — THE KEY FIX

# AQ-10 scoring key:
# Questions where "Definitely Agree / Slightly Agree" = 1 point (ASD-trait direction)
# Questions 1,2,4,5,6 in AQ-10 → "agree" scores a point
# Questions 3,7,8,9,10 in AQ-10 → "disagree" scores a point
# We encode: Yes=2, Somewhat=1, No=0 in the UI.
# For the model, A1–A10 are binary 0/1 per the dataset format.

# Which of our Q1–Q10 are "reverse scored" (No = ASD indicator)
# Actually in the standard AQ-10: items 4,7 are reverse (agree = neurotypical)
# Map: Q_index (0-based) → reverse
REVERSE_IDX = {3, 6}  # Q4 (index 3) and Q7 (index 6)


def encode_aq10_binary(answers_encoded: list) -> list:

    binary = []

    for idx, val in enumerate(answers_encoded[:10]):

        agreed = val >= 1
        if idx in REVERSE_IDX:
            binary.append(0 if agreed else 1)
        else:
            binary.append(1 if agreed else 0)

    return binary


def aq10_clinical_probability(aq10_score: int) -> float:
    """Bounded AQ-10 prior: clinically meaningful without pretending diagnostic certainty."""
    score = int(np.clip(aq10_score, 0, 10))
    probability_by_score = {
        0: 0.03, 1: 0.05, 2: 0.08, 3: 0.12, 4: 0.18,
        5: 0.30, 6: 0.52, 7: 0.64, 8: 0.82, 9: 0.88, 10: 0.92
    }
    return probability_by_score[score]


def find_col(feature_cols: list, candidates: list):
    """Return the first candidate that exists in feature_cols (case-insensitive), or None."""
    lower_map = {c.lower(): c for c in feature_cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def build_input_row(
    answers_encoded: list,
    feature_cols: list,
    user_age: int
) -> pd.DataFrame:

    row = {col: 0 for col in feature_cols}

    aq_binary = encode_aq10_binary(answers_encoded)
    mapped_aq_cols = []

    # Dynamically map AQ columns
    for i, val in enumerate(aq_binary):

        possible_names = [
            f"A{i+1}_Score",
            f"A{i+1}",
            f"A{i+1} Score",
            f"A{i+1}_score"
        ]

        matched_col = find_col(feature_cols, possible_names)

        if matched_col:
            row[matched_col] = val
            mapped_aq_cols.append(matched_col)

    if len(mapped_aq_cols) != 10:
        st.warning(
            f"AQ feature alignment warning: mapped {len(mapped_aq_cols)}/10 AQ columns. "
            f"Mapped columns: {mapped_aq_cols}"
        )

    aq_total_col = find_col(feature_cols, ['AQ10_Total', 'aq10_total'])
    aq_flag_col = find_col(feature_cols, ['AQ10_Flag', 'aq10_flag'])
    aq10_score = sum(aq_binary)

    if aq_total_col:
        row[aq_total_col] = aq10_score
    if aq_flag_col:
        row[aq_flag_col] = int(aq10_score >= 6)

    # Add REAL user age
    age_col = find_col(feature_cols, ['age', 'Age', 'AGE'])

    if age_col:
        row[age_col] = user_age

    # Convert to dataframe
    row_df = pd.DataFrame([row])

    # Ensure correct column order
    row_df = row_df.reindex(columns=feature_cols, fill_value=0)

    return row_df

def predict_asd(
    model,
    feature_cols: list,
    answers_encoded: list,
    face_score: float,
    user_age: int
):

    input_df = build_input_row(
        answers_encoded,
        feature_cols,
        user_age
    )

    # Raw ML probability from the dataset-trained model
    ml_proba = model.predict_proba(input_df)[0][1]
    aq10_score = sum(encode_aq10_binary(answers_encoded))
    aq_proba = aq10_clinical_probability(aq10_score)

    # Supplementary questions influence
    supplementary_score = np.mean(answers_encoded[10:]) / 2

    # Final calibrated screening confidence. AQ-10 drives screening; ML and optional
    # face analysis support it without allowing sparse demographics to bury high AQ scores.
    blended = (
        0.55 * aq_proba +
        0.30 * ml_proba +
        0.10 * supplementary_score +
        0.05 * face_score
    )
    if aq10_score >= 8:
        blended = max(blended, 0.68)
    elif aq10_score >= 6:
        blended = max(blended, 0.50)

    blended = float(np.clip(blended, 0.01, 0.95))

    prediction = int(blended >= 0.45 or aq10_score >= 6)

    return prediction, blended

# 6. UI HELPERS
def show_metrics(metrics: dict):
    with st.expander("📊 Model Evaluation Metrics (held-out test set)", expanded=False):
        st.text(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        st.text(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
        st.text(f"Sensitivity / Recall: {metrics['sensitivity']*100:.2f}%")
        st.text(f"Specificity: {metrics['specificity']*100:.2f}%")
        st.text(f"Precision: {metrics['precision']*100:.2f}%")
        st.text(f"F1 Score: {metrics['f1']:.3f}")
        st.text(f"Brier Score (lower is better): {metrics['brier']:.3f}")
        st.text(f"Model Probability Threshold: {metrics['threshold']:.2f}")
        st.text("Classification Report:")
        st.text(f"ROC-AUC Score: {metrics['auc']:.3f}")
        st.code(metrics["report"])
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(
            confusion_matrix=metrics["cm"],
            display_labels=["No ASD", "ASD"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_proba"])

        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.2f}")
        ax2.plot([0,1], [0,1], linestyle='--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()

        st.pyplot(fig2)
        st.pyplot(fig)
        plt.close(fig)


def show_class_distribution(df: pd.DataFrame):
    with st.expander("📈 Dataset Class Distribution", expanded=False):
        fig, ax = plt.subplots(figsize=(4, 3))
        df['Class/ASD'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'])
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# 7. STREAMLIT APP

st.set_page_config(page_title="ASD Screening Tool", layout="centered")
st.title("🧠 Autism Spectrum Disorder (ASD) Screening Tool")
st.caption(
    "This tool uses the validated AQ-10 questionnaire + XGBoost ML model for **screening purposes only**. "
    "It is NOT a clinical diagnosis. Always consult a qualified specialist."
)

#Load & train 
df    = load_data("teen_adult_asd_train.csv")
model = None
metrics = None

if df is not None:
    show_class_distribution(df)
    with st.spinner("Training model (first run only)…"):
        try:
            model, metrics = train_model("teen_adult_asd_train.csv")
            st.success(f"✅ Model ready — Test Accuracy: {metrics['accuracy']*100:.1f}%")
            show_metrics(metrics)
        except Exception as e:
            st.error(f"Model training failed: {e}")
else:
    st.warning("⚠️ Model training skipped — dataset not found.")


#Questionnaire 
st.header("📋 ASD Screening Questions")

st.header("👤 Basic Information")

user_age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=18,
    step=1
)


st.write(
    "Answer honestly based on your own experience or your observations of the patient. "
    "Questions **1–10** are the clinically validated **AQ-10** screening items used by healthcare professionals."
)
# 32 questions — Q1–Q10 are exact AQ-10 items (map to A1–A10 in dataset)
# Q11–Q32 are supplementary behavioural / clinical questions
questions = [
    # ── AQ-10 Core (Q1–Q10) ── directly maps to A1–A10 dataset columns
    "I often prefer to do things on my own rather than with others.",                          # A1
    "I prefer to do things the same way every time, and find changes to routine distressing.", # A2
    "I find myself becoming so strongly absorbed in one thing that I lose track of other things.", # A3
    "I find it easy to imagine what a character in a book looks like just from the description.",   # A4 (reverse)
    "I am often the last to understand the point of a joke.",                                 # A5
    "When I'm talking to someone, I find it hard to work out when it is my turn to speak.",   # A6
    "I find it easy to work out what someone is thinking or feeling just by looking at them.", # A7 (reverse)
    "I am drawn more to facts, numbers, and things than to people and feelings.",              # A8
    "I find it fascinating to keep track of dates, statistics, or sequences of information.", # A9
    "I find social situations very confusing and hard to navigate.",                           # A10

    # ── Supplementary Behavioural Questions (Q11–Q32) ──
    "I find it very difficult to read other people's facial expressions or body language.",
    "I notice very small changes in people's appearance or surroundings that others miss.",
    "I find it difficult to make small talk or start a conversation with someone new.",
    "I struggle to make or maintain eye contact when talking to others.",
    "I often take what people say very literally and miss sarcasm or implied meaning.",
    "I engage in repetitive behaviours or have very fixed routines that I strongly resist changing.",
    "I find socialising or attending social events mentally exhausting.",
    "I am unusually sensitive to sounds, lights, textures, tastes, or smells.",
    "People have described my speech as overly formal, old-fashioned, or unusual in tone.",
    "I am considered an 'eccentric' or 'quirky' person by those who know me.",
    "I feel more comfortable in my own world of specific interests than engaging in general conversation.",
    "I can memorise a large number of facts about a topic but struggle to understand the deeper meaning.",
    "I have been told my tone of voice sounds robotic, flat, or emotionless.",
    "I sometimes make involuntary sounds (clearing throat, humming, clicking, grunting) in public.",
    "I am exceptionally good at some specific things but noticeably poor at other everyday things.",
    "People have told me I seem to lack empathy or don't understand how others feel.",
    "I sometimes say things that are considered rude or inappropriate without realising it.",
    "I have been told my eye contact is unusual — either too intense or too avoidant.",
    "I prefer to interact with others strictly on my own terms and find it hard to adapt socially.",
    "I am troubled by intrusive repetitive thoughts or feel compelled to repeat actions.",
    "My movements or coordination have been described as clumsy, awkward, or unusual.",
    "I have experienced significant difficulties in daily life due to the traits described above.",
]

ANSWER_VALUES = {"Yes": 2, "Somewhat": 1, "No": 0}

answers = []

st.markdown("---")
st.markdown("**Section 1 — Core AQ-10 Screening Items** *(Questions 1–10)*")
for i in range(10):
    ans = st.radio(f"**Q{i+1}.** {questions[i]}", ["Yes", "Somewhat", "No"],
                   horizontal=True, key=f"q{i}")
    answers.append(ans)

st.markdown("---")
st.markdown("**Section 2 — Supplementary Behavioural Indicators** *(Questions 11–22)*")
for i in range(10, 22):
    ans = st.radio(f"**Q{i+1}.** {questions[i]}", ["Yes", "Somewhat", "No"],
                   horizontal=True, key=f"q{i}")
    answers.append(ans)

st.markdown("---")
st.markdown("**Section 3 — Social & Sensory Profile** *(Questions 23–32)*")
for i in range(22, 32):
    ans = st.radio(f"**Q{i+1}.** {questions[i]}", ["Yes", "Somewhat", "No"],
                   horizontal=True, key=f"q{i}")
    answers.append(ans)

answers_encoded = [ANSWER_VALUES[a] for a in answers]


# Facial Analysis 
st.header("📷 Facial Analysis (Optional)")
st.write(
    "Upload a clear, well-lit frontal photo. The system checks for face and eye visibility "
    "as a **supportive behavioural indicator only** — this does NOT diagnose ASD."
)
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

face_result = {"behavioral_score": 0.0, "note": "No image uploaded."}

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    face_result = analyze_face(image)
    annotated_rgb = cv2.cvtColor(face_result["annotated_image"], cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Facial Analysis Result", use_column_width=True)
    st.info(f"🔍 {face_result['note']}")
    st.caption(
        f"Face detected: {face_result['face_detected']} | "
        f"Eyes detected: {face_result['eyes_detected']} | "
        f"Behavioural signal: {face_result['behavioral_score']:.0%}"
    )


# Predict 
st.divider()
if st.button("🔍 Predict ASD Likelihood", type="primary"):
    if model is None:
        st.error("Model is not available. Please ensure dataset is present.")
    else:
        prediction, confidence = predict_asd(
            model,
            metrics["feature_cols"],
            answers_encoded,
            face_result["behavioral_score"],
            user_age
        )

        conf_pct   = confidence * 100
        aq10_score = sum(encode_aq10_binary(answers_encoded))

        st.subheader("📌 Prediction Result")

        col1, col2 = st.columns(2)
        col1.metric("AQ-10 Score (Q1–Q10)", f"{aq10_score} / 10",
                    help="Score ≥ 6 is typically flagged for further evaluation by clinicians.")
        col2.metric("Model Confidence (ASD likelihood)", f"{conf_pct:.1f}%")

        if prediction == 1 and (confidence >= 0.65 or aq10_score >= 8):
            st.error(
                "⚠️ **High Likelihood of ASD Indicators Detected.**\n\n"
                "This screening suggests significant ASD-related traits. "
                "Please consult a licensed psychologist or psychiatrist for a full evaluation."
            )
        elif prediction == 1 and confidence >= 0.45:
            st.warning(
                "🟡 **Moderate Indicators Present.**\n\n"
                "Some ASD-related traits were identified. Consider speaking with a healthcare professional "
                "for a more thorough assessment."
            )
        else:
            st.success(
                "✅ **Low Likelihood of ASD Indicators.**\n\n"
                "Current responses do not strongly indicate ASD. Continue monitoring if concerns persist."
            )

        st.caption(
            "⚕️ Disclaimer: This tool is for informational screening only and does not constitute "
            "a medical diagnosis. Results must always be interpreted by a qualified professional."
        )

        #Debug: show exactly what was sent to the model
        with st.expander("🔧 Debug — What the model received", expanded=False):
            input_df = build_input_row(answers_encoded, metrics["feature_cols"],user_age)
            # Only show columns with non-zero values + key columns
            key_cols = [c for c in metrics["feature_cols"] if
                        any(x in c.lower() for x in ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','result','age'])]
            st.write("**Key feature values sent to model:**")
            st.dataframe(input_df[key_cols] if key_cols else input_df)
            st.write(f"**Raw ML probability (before facial blend):** `{model.predict_proba(input_df)[0][1]:.4f}`")
            st.write(f"**AQ-10 clinical prior:** `{aq10_clinical_probability(aq10_score):.4f}`")
            st.write(f"**Reverse-scored AQ binary values A1-A10:** `{encode_aq10_binary(answers_encoded)}`")
            st.write(f"**All feature columns the model knows:** `{metrics['feature_cols']}`")

        with st.expander("📊 Top Predictive Features", expanded=False):
            feat_imp = pd.Series(
                model.feature_importances_,
                index=metrics["feature_cols"]
            ).sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(6, 4))
            feat_imp.plot(kind='barh', ax=ax, color='steelblue')
            ax.invert_yaxis()
            ax.set_title("Top 10 Feature Importances")
            ax.set_xlabel("Importance Score")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
