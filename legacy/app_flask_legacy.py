# Reload trigger
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    coefs = model.coef_[0].tolist()
    features_names = ['Age', 'Gender', 'Height', 'Weight', 'Sys BP', 'Dia BP', 'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active']
    return render_template('index.html', coefs=coefs, features=features_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['height']),
            float(request.form['weight']),
            float(request.form['ap_hi']),
            float(request.form['ap_lo']),
            float(request.form['cholesterol']),
            float(request.form['gluc']),
            float(request.form['smoke']),
            float(request.form['alco']),
            float(request.form['active'])
        ]
        
        # Scale features
        final_features = scaler.transform([np.array(features)])
        
        # Predict
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0][1] * 100 # Risk %
        
        # Format output
        output = 'High Risk of Heart Disease' if int(prediction[0]) == 1 else 'Low Risk of Heart Disease'
        
        coefs = model.coef_[0].tolist()
        features_names = ['Age', 'Gender', 'Height', 'Weight', 'Sys BP', 'Dia BP', 'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active']
        
        return render_template('index.html', prediction_text=output, probability=round(probability, 2), coefs=coefs, features=features_names)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", coefs=[], features=[])

if __name__ == "__main__":
    app.run(debug=True)
