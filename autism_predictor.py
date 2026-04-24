import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_data(df):
    df = df.drop(['ID', 'contry_of_res', 'age_desc', 'relation'], axis=1)
    categorical_columns = ['gender', 'ethnicity', 'jaundice', 'austim', 'used_app_before']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    X = df.drop('Class/ASD', axis=1)
    y = df['Class/ASD']
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled
    #return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #model = RandomForestClassifier(random_state=42)
    #model.fit(X_train, y_train)
    params = {
        'n_estimators': [500],
        'max_depth': [5],
        'learning_rate': [0.05],
        'subsample': [0.85], 
        'colsample_bytree': [0.75],
        'reg_lambda': [0.2],  
        'reg_alpha': [0.1] 
        
    }

    grid = GridSearchCV(XGBClassifier(), param_grid=params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Best Parameters:", grid.best_params_) 
    return best_model, accuracy
    #return model, accuracy
    


def analyze_face(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    
    if len(faces) > 0:
        return 1  


st.title("Autism Predictor with Machine Learning")


st.header("ASD Screening Questions")
st.write("Instructions:")
st.write("* This Screening Test assesses various behaviours and communication skills associated with Autism Spectrum Disorder(ASD).")
st.write("* Please complete this form based on your observations and interactions with the patient.")
questions = [
    "Do you  prefer to do things on your own, rather than with others?",
    "Do you prefer doing things the same way - for instance your morning routine or trip to the supermarket?",
    "Do you find yourself becoming strongly absorbed in something - even obsessional?",
    "Do you  find it easy to imagine what characters from a book might look like?",
    "Are you very sensitive to noise and will wear earplugs or cover your ears in certain situations?",
    "Do sometimes people say that you are being rude, even though you think you are being polite?",
    "Do you find it easy to talk in groups of people?",
    "Are you more interested in finding out about 'things' than people?",
    "Do you find numbers, dates and strings of information fascinating?",
    "Do you find it upsetting if your daily routine is upset or changed?",
    "Is it difficult for you to understand other people's facial expression and body language?",
    "Do you notice very small changes in a person's appearance?",
    "Do you have any problems making small talk with new people?",
    "Do you have difficulty making eye contact?",
    "Do you find it difficult to understand what others mean?",
    "Is there repetitive or stereotyped language or behaviour?",
    "Do you find it hard to socialize?",
    "Do you have sensory sensitivities?",
    "Please share how your behaviour stands out different from others of your age: old fashioned or precocious?",
    "Please share how your behaviour stands out different from others of your age: regarded as an 'eccentric professor' by the other people?",
    "Please share how your behaviour stands out different from others of your age: lives somewhat in a world of his/her own with restricted idiosyncratic intellectual interests?",
    "Please share how your behaviour stands out different from others of your age: accumulates facts on certain subjects (good rote memory) but does not really understand the meaning?",
    "Please share how your behaviour stands out different from others of your age: has a deviant style of communication with a formal, fussy, old-fashioned or 'robot like' language?",
    "Please share how your behaviour stands out different from others of your age: expresses sounds involuntarily; clears throat, grunts, smacks, cries or screams?",
    "Please share how your behaviour stands out different from others of your age: is surprisingly good at some things and surprisingly poor at others?",
    "Please share how your behaviour stands out different from others of your age: lacks empathy?",
    "Please share how your behaviour stands out different from others of your age: makes naive and embarrassing remarks?",
    "Please share how your behaviour stands out different from others of your age: has a deviant style of gaze?",
    "Please share how your behaviour stands out different from others of your age: can be with other people but only on your terms?",
    "Please share how your behaviour stands out different from others of your age: has difficulties in completing simple daily activities because of compulsory repetition of certain actions or thoughts?",
    "Please share how your behaviour stands out different from others of your age: has clumsy, ill coordinated, ungainly, awkward movements or gestures?"
]

answers = []
for question in questions:
    answers.append(st.radio(question, ["Yes", "Somewhat", "No"]))
answer_values = {"Yes":2, "No":0, "Somewhat":1}
answers_encoded = [answer_values[ans] for ans in answers]


st.header("Facial Analysis (Optional)")
uploaded_file = st.file_uploader("Upload a photo for analysis", type=['jpg', 'png', 'jpeg'])
face_score = 0

if uploaded_file:
  
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    face_score = analyze_face(image)


final_score = (sum(answers_encoded)/31) * 0.7 + face_score * 0.3


threshold = 1  

if st.button("Predict Autism (ASD)") :
    if final_score > threshold:
        st.error("High Likelihood of ASD. Consult a Specialist.")
    elif final_score > threshold/2:
        st.success("Low Likelihood of ASD. Continue Monitoring.")
    else:
        st.success("You are normal for now.")

def load_data():
    try:
        dataset_path = "train.csv"  
        df = pd.read_csv(dataset_path)
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        df['Class/ASD'].value_counts().plot(kind='bar', ax=ax, title="Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        st.pyplot(fig)
        return df
    except FileNotFoundError:
        st.error("Dataset file 'train.csv' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


df = load_data()

if df is not None:
    try:
       
        X, y = preprocess_data(df)

        
        model, accuracy = train_model(X, y)
        st.text(f"Model trained with {accuracy*100:.2f}% accuracy")
    except Exception as e:
        st.error(f"Error during model training: {e}")
else:
    st.warning("Model training skipped due to dataset loading error.")
