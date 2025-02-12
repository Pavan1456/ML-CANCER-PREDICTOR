import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def load_and_prepare_data(csv_path: str):

    data = pd.read_csv(csv_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    
    # Separate features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis'].map({'M': 1, 'B': 0})
    return X, y

# @st.cache_resource(show_spinner=False)
def train_classification_models(X_train_scaled, y_train, X_test_scaled, y_test):

    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=10000),
        'Naive Bayes': GaussianNB()
    }
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        model_scores[name] = (model, accuracy)
    return model_scores



def main():
    st.set_page_config(
        page_title="Cancer Diagnosis Predictor", 
        layout="wide", 
        page_icon="🩺"
    )
    st.title("🩺 Cancer Diagnosis Predicting...")

    X, y = load_and_prepare_data('D:\workshop\MLproject\cancer.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_scores = train_classification_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Sidebar: Instructions, Model Selection, and Accuracy Display
    with st.sidebar:
        st.header(":orange[PAWAN AGRAWAL]")
        st.header(":blue[🔹 Instructions]")
        st.markdown(
            """
            1. Enter the values for each feature below.
            2. Choose the model to make the prediction.
            3. Click *Predict* to see the diagnosis result.
            """
        )

        st.header(":globe_with_meridians: Select Model")
        selected_model_name = st.selectbox(
            "Choose Classification Model", 
            list(model_scores.keys())
        )

        st.header(":dart: Model Accuracies")
        for model_name, (_, accuracy) in model_scores.items():
            st.write(f"*{model_name}:* {accuracy * 100:.2f}%")


    st.markdown("""
    <style>
        .stButton>button {
            background-color: #6200EA;
            color: white;
            padding: 12px 30px;
            border-radius: 20px;
            border: none;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background-color: #3700B3;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            border-radius: 10px;
            padding: 10px;
        }
        .stNumberInput label {
            color: #6200EA;
            font-weight: bold;
            font-size: 14px;
        }
        .stForm {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Main Content: Patient Data Input

    st.header(":calendar: Patient Data 👨‍⚕️")

    # Group features based on their suffix in the column names
    mean_features = [feature for feature in X.columns if 'mean' in feature]
    se_features = [feature for feature in X.columns if 'se' in feature]
    worst_features = [feature for feature in X.columns if 'worst' in feature]

    input_data = {}

    # Create a form for user input
    with st.form(key='input_form'):
        cols = st.columns(3)
        with cols[0]:
            st.subheader(":orange[Mean Features]")
            for feature in mean_features:
                default_value = float(X[feature].mean())
                input_data[feature] = st.number_input(
                    label=feature,
                    value=default_value,
                    step=0.01,
                    format="%0.2f"
                )
        with cols[1]:
            st.subheader(":orange[Standard Error Features]")
            for feature in se_features:
                default_value = float(X[feature].mean())
                input_data[feature] = st.number_input(
                    label=feature,
                    value=default_value,
                    step=0.01,
                    format="%0.2f"
                )
        with cols[2]:
            st.subheader(":orange[Worst Features]")
            for feature in worst_features:
                default_value = float(X[feature].mean())
                input_data[feature] = st.number_input(
                    label=feature,
                    value=default_value,
                    step=0.01,
                    format="%0.2f"
                )
        submit_button = st.form_submit_button(label='Predict Diagnosis')

    # Prediction and Result Display
    if submit_button:
        selected_model = model_scores[selected_model_name][0]
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        prediction = selected_model.predict(input_scaled)[0]
        diagnosis = 'Malignant (M)' if prediction == 1 else 'Benign (B)'
        result_color = '#FF4B4B' if prediction == 1 else '#4CAF50'
        
        # Display the result in a styled container
        st.markdown(f"""
        <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: #333;'>Diagnosis Result</h2>
            <h1 style='color: {result_color};'>{diagnosis}</h1>
            <p style='color: #555;'>
                Model Used: <strong>{selected_model_name}</strong> 
                (Accuracy: {model_scores[selected_model_name][1] * 100:.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
