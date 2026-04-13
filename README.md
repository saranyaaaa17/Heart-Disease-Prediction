# Heart Disease Prediction Platform 🚀

An advanced, enterprise-grade machine learning application that predicts the risk of cardiovascular disease based on clinical metrics. Built entirely with a decoupled robust backend and cutting-edge React frontend architecture.

## 🌟 Features
- **Accurate Medical Predictions**: Trained on 70,000 clinically validated patient records (OpenML Dataset ID 45547) via Logistic Regression.
- **Model Explainability**: Dynamic real-time charts illustrating feature correlations (Systolic BP, Cholesterol, Age, Weight, etc.) powered by `Chart.js`.
- **FastAPI Engine**: Lightning fast, strictly typed REST API backend.
- **Premium Dashboard UI**: React.js based dashboard utilizing a dark-mode, glassmorphism aesthetic built completely with Vanilla CSS.

## 🛠️ Architecture Stack
- **AI/ML**: `scikit-learn`, `pandas`, `numpy`
- **Backend API**: `FastAPI`, `uvicorn`, `pydantic`
- **Frontend App**: `React`, `Vite`, `axios`, `react-chartjs-2`

## ⚙️ Running Locally

### 1. Initialize the Backend
1. Make sure Python is installed.
2. Install model dependencies:
   ```bash
   pip install fastapi uvicorn pydantic flask-cors scikit-learn pandas numpy
   ```
3. *(Optional)* Retrain the model on the latest dataset:
   ```bash
   python train.py
   ```
4. Start the inference engine:
   ```bash
   python main.py
   ```
   > The API will open on `http://127.0.0.1:8000`.

### 2. Initialize the Web App
1. Open a new terminal.
2. Navigate to the `frontend` folder.
   ```bash
   cd frontend
   npm install
   ```
3. Run the development environment:
   ```bash
   npm run dev
   ```
   > Access the live UI at `http://localhost:5173`.
