# Autism Predictor with Machine Learning


## 🚀 Overview
This project is a **Machine Learning-based Autism Predictor** designed to assess the likelihood of Autism Spectrum Disorder (ASD) based on user responses and optional facial analysis. It employs **XGBoost Classifier with GridSearchCV** for hyperparameter tuning and **SMOTE** for handling imbalanced datasets. The application is built with **Streamlit**, providing an interactive and user-friendly interface for ASD screening.


## 💻Demo Video

https://drive.google.com/file/d/1YQWXFSXtlV8HbRmffKJMPws130cBUeLq/view?usp=sharing

## 🌟 Features
✅ **Dual-Layer Questionnaire System** – Integrates the 10 clinically validated AQ-10 core screening items with 22 supplementary behavioral profile questions.

✅ **Computer Vision Gaze Analysis (Optional)** – Utilizes OpenCV Haar Cascades to analyze frontal face and eye geometry, flagging partial or avoidant eye contact as a supportive behavioral signal.

✅ **Mathematical Prediction Blending** – Implements a calibrated confidence algorithm that mathematically weights clinical AQ-10 priors (55%), raw ML probabilities (30%), supplementary features (10%), and vision signals (5%).

✅ **Overfitting Mitigated Architecture** – Employs structural regularization (shallow tree depth, feature/subsample constraints) and native algorithmic class weighting (`scale_pos_weight`) to handle imbalanced training data robustly.

✅ **Real-Time Evaluation Metrics** – Features an expandable runtime diagnostics panel displaying accuracy, sensitivity, specificity, a live confusion matrix, and ROC-AUC charting directly on the dashboard.

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/autism-predictor.git
   cd autism-predictor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
- The model is trained on an **Autism Spectrum Disorder (ASD) Screening dataset** (e.g., `train.csv`).
- The dataset includes key features such as **age, gender, ethnicity, screening test responses, and ASD diagnosis**.
- Data preprocessing includes handling missing values, dropping redundant columns, and encoding categorical features.

## 🧠 Model & Training
- **Preprocessing:** The dataset is cleaned by removing irrelevant features and encoding categorical variables.
- **Oversampling:** **SMOTE (Synthetic Minority Over-sampling Technique)** is used to address class imbalance.
- **Classifier:** The model leverages **XGBoost**, known for its high efficiency and performance in classification tasks.
- **Hyperparameter Tuning:** **GridSearchCV** optimizes parameters such as learning rate, max depth, and number of estimators.
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

