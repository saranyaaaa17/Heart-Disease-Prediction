import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

def train_and_save_model():
    # Load real medical dataset
    print("Fetching OpenML Cardiovascular Disease dataset (ID: 45547)...")
    data = fetch_openml(data_id=45547, parser='auto')
    df = data.frame
    
    # Drop rows with NaN if any exist (usually none in this set)
    df = df.dropna()
    
    # Convert age from days to years
    df['age'] = (df['age'] / 365).astype(int)
    
    # Target and features
    X = df.drop('cardio', axis=1)
    y = df['cardio'].astype(int)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model trained successfully. Features: {list(X.columns)}")
    
    # Save the model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Model and scaler saved to model.pkl and scaler.pkl")

if __name__ == '__main__':
    train_and_save_model()
