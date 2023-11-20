import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify 

import os

app = Flask(__name__)

#

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
# Route to serve the chatbot.html file
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
# Route for home.html
@app.route('/home')
def home_page():
    return render_template('home.html', feature_names=feature_names)




@app.route('/')
def index():
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


# Rule-based breast cancer chatbot function
def get_response(user_input):
    user_input = user_input.lower()
    if 'breast cancer' in user_input:
        return "Breast cancer is a type of cancer that forms in the cells of the breasts. It can occur in both men and women, but it's far more common in women."
    elif 'symptoms' in user_input:
        return "Common symptoms of breast cancer may include a lump in the breast, changes in breast size or shape, skin dimpling, nipple inversion, or nipple discharge."
    elif 'risk factors' in user_input:
        return "Risk factors for breast cancer include age, family history, genetic mutations (BRCA genes), obesity, alcohol consumption, and hormone replacement therapy."
    elif 'diagnosis' in user_input:
        return "Breast cancer diagnosis involves mammograms, biopsies, MRI, and other imaging tests to detect abnormalities and confirm the presence of cancerous cells."
    elif 'treatment' in user_input:
        return "Treatment options for breast cancer may include surgery, chemotherapy, radiation therapy, hormone therapy, targeted therapy, or a combination of these."
    elif 'prevention' in user_input:
        return "To lower the risk of breast cancer, maintain a healthy lifestyle, limit alcohol intake, exercise regularly, maintain a healthy weight, and consider regular screenings."
    else:
        return "I'm sorry, I might not have information on that specific topic related to breast cancer. Please ask another question."


@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']

    if not user_message:
        return jsonify({'bot_response': 'Please provide a valid input.'})

    bot_response = get_response(user_message)
    return jsonify({'bot_response': bot_response})


if __name__ == '__main__':
    app.run(debug=True)

