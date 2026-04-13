import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './index.css'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    height: '',
    weight: '',
    ap_hi: '',
    ap_lo: '',
    cholesterol: '1',
    gluc: '1',
    smoke: '0',
    alco: '0',
    active: '1'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [explainability, setExplainability] = useState(null);

  // Fetch model coefficients on mount
  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        const res = await axios.get('http://127.0.0.1:8000/features');
        setExplainability(res.data);
      } catch (err) {
        console.error("Failed to load model features", err);
      }
    };
    fetchFeatures();
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    // Ensure all numeric properly
    const payload = Object.keys(formData).reduce((acc, key) => {
      acc[key] = parseFloat(formData[key]);
      return acc;
    }, {});

    try {
      const res = await axios.post('http://127.0.0.1:8000/predict', payload);
      setResult(res.data);
      // Reset form visually
      setFormData({
        age: '', gender: '', height: '', weight: '', ap_hi: '', ap_lo: '',
        cholesterol: '1', gluc: '1', smoke: '0', alco: '0', active: '1'
      });
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const renderChart = () => {
    if (!explainability) return null;

    const { features, coefficients } = explainability;
    const maxAbs = Math.max(...coefficients.map(Math.abs));
    
    const colors = coefficients.map(c => {
      const intensity = 0.3 + (Math.abs(c) / maxAbs) * 0.7;
      return c >= 0 ? `rgba(239, 68, 68, ${intensity})` : `rgba(16, 185, 129, ${intensity})`;
    });

    const data = {
      labels: features,
      datasets: [
        {
          label: 'Impact on Heart Disease Risk',
          data: coefficients,
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (context) => context[0].label,
            label: (context) => {
              const val = context.raw;
              return (val > 0 ? "Increases risk by: " : "Reduces risk by: ") + Math.abs(val).toFixed(3);
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1' }
        }
      }
    };

    return <Bar data={data} options={options} />;
  };

  return (
    <div className="app-wrapper">
      <header className="header">
        <h1>CardioAI Analytics</h1>
        <p style={{ color: "var(--text-muted)", fontSize: "1.2rem" }}>Advanced Clinical Decision Support System</p>
      </header>

      <div className="main-grid">
        
        {/* Left: Input Form */}
        <div className="glass-card">
          <h2 className="card-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
            Patient Data Entry
          </h2>
          
          <form className="form-grid" onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Age (Yrs)</label>
              <input type="number" name="age" value={formData.age} onChange={handleChange} required min="1" max="120" placeholder="e.g. 52"/>
            </div>
            
            <div className="form-group">
              <label>Gender</label>
              <select name="gender" value={formData.gender} onChange={handleChange} required>
                <option value="" disabled>Select</option>
                <option value="1">Male</option>
                <option value="2">Female</option>
              </select>
            </div>

            <div className="form-group">
              <label>Height (cm)</label>
              <input type="number" name="height" value={formData.height} onChange={handleChange} required placeholder="170"/>
            </div>

            <div className="form-group">
              <label>Weight (kg)</label>
              <input type="number" name="weight" value={formData.weight} onChange={handleChange} required step="0.1" placeholder="75.5"/>
            </div>

            <div className="form-group">
              <label>Systolic BP</label>
              <input type="number" name="ap_hi" value={formData.ap_hi} onChange={handleChange} required placeholder="120"/>
            </div>

            <div className="form-group">
              <label>Diastolic BP</label>
              <input type="number" name="ap_lo" value={formData.ap_lo} onChange={handleChange} required placeholder="80"/>
            </div>

            <div className="form-group">
              <label>Cholesterol</label>
              <select name="cholesterol" value={formData.cholesterol} onChange={handleChange} required>
                <option value="1">Normal</option>
                <option value="2">Borderline</option>
                <option value="3">High</option>
              </select>
            </div>

            <div className="form-group">
              <label>Glucose</label>
              <select name="gluc" value={formData.gluc} onChange={handleChange} required>
                <option value="1">Normal</option>
                <option value="2">Borderline</option>
                <option value="3">High</option>
              </select>
            </div>

            <div className="form-group">
              <label>Smoker</label>
              <select name="smoke" value={formData.smoke} onChange={handleChange} required>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label>Alcohol Use</label>
              <select name="alco" value={formData.alco} onChange={handleChange} required>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="form-group full-width">
              <label>Physical Activity Profile</label>
              <select name="active" value={formData.active} onChange={handleChange} required>
                <option value="1">Active (Regularly Exercises)</option>
                <option value="0">Sedentary (Little/No Exercise)</option>
              </select>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? <span className="spinner"></span> : 'Execute Risk Analysis'}
            </button>
          </form>
        </div>

        {/* Right: Results & Insights */}
        <div className="glass-card">
          <h2 className="card-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            ML Assessment Engine
          </h2>

          <div className="result-display">
            {result ? (
              <>
                <div className={`confidence-circle ${result.is_high_risk ? 'danger' : 'safe'}`}>
                  <div className="confidence-value">{result.risk_probability}%</div>
                  <div className="confidence-label">Risk</div>
                </div>
                <div className={`result-status ${result.is_high_risk ? 'status-danger' : 'status-safe'}`}>
                  {result.prediction}
                </div>
                <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                  {result.is_high_risk 
                    ? "Clinical intervention advised based on elevated ML risk scorings."
                    : "Patient metrics indicate nominal risk parameters."}
                </p>
              </>
            ) : (
              <div style={{ padding: "3rem 0", color: "var(--text-muted)" }}>
                Awaiting input parameters for inference.
              </div>
            )}
          </div>

          <div style={{ marginTop: "2rem" }}>
            <h3 style={{ fontSize: "1.1rem", marginBottom: "0.5rem", fontWeight: "600" }}>Feature Importance Map</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
              Global logistic regression coefficients indicating impact weights.
            </p>
            <div className="chart-container">
              {explainability ? renderChart() : <span className="spinner"></span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
