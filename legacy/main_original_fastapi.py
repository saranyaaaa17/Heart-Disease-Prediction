import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

class PatientMetrics(BaseModel):
    age: float
    gender: float
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: float
    gluc: float
    smoke: float
    alco: float
    active: float

@app.get("/features")
def get_features():
    coefs = model.coef_[0].tolist()
    features_names = ['Age', 'Gender', 'Height', 'Weight', 'Sys BP', 'Dia BP', 'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active']
    return {"features": features_names, "coefficients": coefs}

@app.post("/predict")
def predict(metrics: PatientMetrics):
    try:
        features = [
            metrics.age,
            metrics.gender,
            metrics.height,
            metrics.weight,
            metrics.ap_hi,
            metrics.ap_lo,
            metrics.cholesterol,
            metrics.gluc,
            metrics.smoke,
            metrics.alco,
            metrics.active
        ]
        
        # Scale features
        final_features = scaler.transform([np.array(features)])
        
        # Predict
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0][1] * 100
        
        output = 'High Risk of Heart Disease' if int(prediction[0]) == 1 else 'Low Risk of Heart Disease'
        
        return {
            "prediction": output,
            "risk_probability": round(probability, 2),
            "is_high_risk": bool(int(prediction[0]) == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
