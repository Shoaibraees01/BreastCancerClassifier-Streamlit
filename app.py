import streamlit as st
import pandas as pd
import joblib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import base64

# Define all features (from the dataset)
all_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Selected features (from notebook's SelectKBest)
selected_features = [
    'mean radius', 'mean perimeter', 'mean area', 'mean concavity',
    'mean concave points', 'worst radius', 'worst perimeter', 'worst area',
    'worst concavity', 'worst concave points'
]

# Class names
class_names = ["Malignant", "Benign"]

# Set page configuration for styling
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    body {
        background-color: #fce4ec; /* Light pink background */
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9); /* Slightly transparent white for contrast */
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: 0 auto;
    }
    .header {
        color: #d32f2f; /* Pink shade for breast cancer awareness */
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .subheader {
        color: #333;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .stNumberInput > div {
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #b71c1c;
    }
    .download-btn {
        background-color: #4caf50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin-top: 1rem;
    }
    .download-btn:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<div class="header">Breast Cancer Classification</div>', unsafe_allow_html=True)

# Select classifier
classifier = st.selectbox("Choose Classifier", ["Logistic Regression", "Random Forest"], key="classifier_select")

# Toggle optimization
optimization = st.radio("Optimization", ["Without Feature Selection", "With Feature Selection"], key="optimization_radio")

# Determine model path and features based on selections
if classifier == "Logistic Regression":
    if optimization == "Without Feature Selection":
        model_path = "models/logistic_regression_without.pkl"
        features = all_features
    else:
        model_path = "models/logistic_regression_with.pkl"
        features = selected_features
else:  # Random Forest
    if optimization == "Without Feature Selection":
        model_path = "models/random_forest_without.pkl"
        features = all_features
    else:
        model_path = "models/random_forest_with.pkl"
        features = selected_features

# Load the selected model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}. Ensure models are saved from the notebook.")
    st.stop()

# Input form for features
st.markdown('<div class="subheader">Enter Feature Values</div>', unsafe_allow_html=True)
inputs = {}
for feature in features:
    inputs[feature] = st.number_input(feature, value=0.0, step=0.01, key=f"{feature}_input")

# Predict button
if st.button("Predict", key="predict_button"):
    # Convert inputs to NumPy array without feature names
    input_data = np.array(list(inputs.values())).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Display results
    st.markdown('<div class="subheader">Prediction Results</div>', unsafe_allow_html=True)
    st.write(f"Predicted Class: **{class_names[prediction]}**")
    
    st.markdown('<div class="subheader">Probability Scores</div>', unsafe_allow_html=True)
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: **{prob:.4f}**")

    # Generate PDF report in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    report_content = [
        Paragraph(f"<b>Breast Cancer Classification Report</b>", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"<b>Classifier Used:</b> {classifier}", styles['Normal']),
        Paragraph(f"<b>Optimization:</b> {optimization}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("<b>Patient Feature Values:</b>", styles['Heading2'])
    ]
    for feature, value in inputs.items():
        report_content.append(Paragraph(f"{feature}: {value}", styles['Normal']))
    report_content.extend([
        Spacer(1, 12),
        Paragraph("<b>Prediction Results:</b>", styles['Heading2']),
        Paragraph(f"Predicted Class: {class_names[prediction]}", styles['Normal']),
        Paragraph(f"Probability of Malignant: {probabilities[0]:.4f}", styles['Normal']),
        Paragraph(f"Probability of Benign: {probabilities[1]:.4f}", styles['Normal'])
    ])
    doc.build(report_content)
    pdf_data = buffer.getvalue()
    buffer.close()

    # Provide download link
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="patient_report.pdf" class="download-btn">Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)