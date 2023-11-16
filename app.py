import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from google.cloud import dialogflow
import os

app = Flask(__name__)

# Set your Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"sharp-fire-405218-3c0886101406.json"

# Load the pre-trained model
model = pickle.load(open('cancer_model.pkl', 'rb'))

# Feature names for scaling
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Fetch the form data
        data = {key: float(request.form[key]) for key in feature_names}

        # Convert the input data into a format that can be used for prediction
        input_features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = model.predict(input_features)
        
        # Map the prediction value to 'benign' or 'malignant'
        diagnosis_mapping = {0: 'benign', 1: 'malignant'}
        diagnosis = diagnosis_mapping[prediction[0]]

        # Process the prediction as needed
        # For example, you might want to display the result on a new page
        return render_template('result.html', prediction=diagnosis)

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']

    # Check if user_message is empty or None
    if not user_message:
        return jsonify({'bot_response': 'Please provide a valid input.'})

    session_id = "unique-session-id"  # You can generate a unique session ID
    project_id = "sharp-fire-405218"   # Replace with your Dialogflow project ID

    client = dialogflow.SessionsClient()
    session = client.session_path(project_id, session_id)

    text_input = dialogflow.TextInput(text=user_message, language_code="en")
    query_input = dialogflow.QueryInput(text=text_input)

    response = client.detect_intent(
        request={"session": session, "query_input": query_input}
    )

    bot_response = response.query_result.fulfillment_text
    return jsonify({'bot_response': bot_response})

@app.route('/chatbot')  # Endpoint for serving chatbot.html
def chatbot():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
