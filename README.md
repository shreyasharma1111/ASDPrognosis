# Autism Predictor with Machine Learning


## 🚀 Overview
ASDPrognosis is a Machine Learning-based screening tool designed to assess the likelihood of Autism Spectrum Disorder (ASD) indicators using a multi-modal approach. The system processes a clinically validated 32-question behavioural assessment paired with an optional computer vision facial/gaze analysis module. 

The backend utilises an optimized, regularized **XGBoost Classifier** configured natively to handle real-world dataset class imbalances without synthetic data generation, preventing model overfitting and ensuring strong generalisation. The interactive frontend is deployed via **Streamlit**.

## 🌟 Features
- **Dual-Layer Questionnaire System** – Integrates the 10 clinically validated AQ-10 core screening items with 22 supplementary behavioral profile questions.

- **Computer Vision Gaze Analysis (Optional)** – Utilizes OpenCV Haar Cascades to analyse frontal face and eye geometry, flagging partial or avoidant eye contact as a supportive behavioral signal.

- **Mathematical Prediction Blending** – Implements a calibrated confidence algorithm that mathematically weights clinical AQ-10 priors (55%), raw ML probabilities (30%), supplementary features (10%), and vision signals (5%).

- **Overfitting Mitigated Architecture** – Employs structural regularization (shallow tree depth, feature/subsample constraints) and native algorithmic class weighting (`scale_pos_weight`) to handle imbalanced training data robustly.

- **Real-Time Evaluation Metrics** – Features an expandable runtime diagnostics panel displaying accuracy, sensitivity, specificity, a live confusion matrix, and ROC-AUC charting directly on the dashboard.

**NOTE:**   This tool is intended for **screening support only** and is **not a medical diagnosis system**.

Facial analysis contributes only a small calibrated weight to the final prediction.

---

## 📊 Model Performance

| Metric               | Score  |
| -------------------- | ------ |
| Accuracy             | 96.91% |
| Balanced Accuracy    | 97.22% |
| Recall / Sensitivity | 98.04% |
| Specificity          | 96.40% |
| Precision            | 92.59% |
| F1 Score             | 0.952  |
| ROC-AUC              | 0.999  |
| Brier Score          | 0.021  |

---
## 📂 Dataset Sources

This project was trained using clinically inspired AQ-10 ASD screening datasets from the UCI Machine Learning Repository:

- [Adult ASD Screening Dataset](https://archive.ics.uci.edu/dataset/426/autism+screening+adult)

- [Adolescent ASD Screening Dataset](https://archive.ics.uci.edu/dataset/420/autistic+spectrum+disorder+screening+data+for+adolescent)

---
## 🚀 Live Demo

🔗 https://asdprognosis.streamlit.app/


## 💻Demo Video

https://drive.google.com/file/d/1YQWXFSXtlV8HbRmffKJMPws130cBUeLq/view?usp=sharing

---

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shreyasharma1111/ASDPrognosis.git
   cd ASDPrognosis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run autism_predictor_refined.py
   ```

---
## 🧠 Model & Training
- **Preprocessing:**
  * Missing value handling

  * Dynamic AQ-10 column mapping
  
  * One-hot encoding
  
  * Feature alignment between train/test datasets
- **Oversampling:** Uses data-driven calculated negative-to-positive ratio weights applied directly inside the tree-split mechanics to prevent majority-class bias natively.
- **Classifier:** The model utilises **XGBoost**, known for its high efficiency and performance in classification tasks.
- **Model Regularization:** Parameters such as learning rate, max depth, and number of estimators were manually tuned to constrain tree complexity and prevent overfitting.
- **Threshold Tuning:** Calibrated at an empirical prediction threshold of 0.50 to maintain high sensitivity and balanced accuracy on unseen holdout test splits.
- **Evaluation:** The model reports the best accuracy score after training and tuning.

## 🎯 Usage
1. **Answer the ASD screening questionnaire by selecting responses to behavioral questions.**
2. **(Optional) Upload a facial image for analysis.**
3. **Click the 'Predict Autism (ASD)' button.**
4. **The app calculates a final ASD likelihood score using questionnaire responses and facial analysis.**
5. **Results are displayed with recommendations for further screening if necessary.**

## 🤝 Contributing
We welcome contributions from the community! Follow these steps to contribute:
1. **Fork the repository.**
2. **Create a new branch:**
   ```bash
   git checkout -b feature-branch
   ```
3. **Make your changes and commit them:**
   ```bash
   git commit -m "Add new feature"
   ```
4. **Push to the branch and create a Pull Request.**

## 🚀 Future Enhancements
🔹 Improve facial analysis by integrating **deep learning models for feature extraction**.

🔹 Expand the dataset to include **more diverse populations** for better generalization.

🔹 Enhance the web app with **real-time feedback and report generation**.

🔹 Deploy the application using **cloud services** for wider accessibility.



## 📧 Contact
For any queries, feel free to open an issue on **[GitHub Issues](https://github.com/yourusername/autism-predictor/issues)** or reach out via email.

