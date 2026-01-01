# 🩺 Smart Health Monitoring & Risk Prediction System

## 🎯 Project Overview
*CSE Final Year Project 5* - IoT wearable data → ML Risk Classification → 3D Dashboard

*Problem Solved*: Reactive health monitoring → Predictive risk alerts using continuous physiological signals (Heart Rate, SpO2, Activity)

## 🛠 Tech Stack
- **Frontend**: Streamlit (Interactive Dashboard)
- **Backend**: FastAPI (Risk Prediction API)
- **ML**: Scikit-learn (Logistic Regression for Risk Classification)
- **Visualization**: Plotly (Real-time Charts), PyVista (3D Heart Model)
- **Data**: Pandas, NumPy (Simulated Health Data)

## 🚀 Features
- Real-time health monitoring dashboard
- 3D interactive heart visualization
- Risk prediction using ML (Normal/Warning/Critical)
- Live data streaming simulation
- FastAPI backend for API endpoints
- Responsive UI with custom CSS

## 📋 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-health-monitor.git
   cd smart-health-monitor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. (Optional) Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## 🌐 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and set main file to `app.py`
5. Deploy!

### Local Deployment
- Use the installation steps above
- For production, consider Docker or cloud platforms

## 📊 API Endpoints
- `POST /predict`: Predict risk level
  - Body: `{"heart_rate": 75, "spo2": 98, "activity_level": 50}`
  - Response: `{"risk_level": "Normal", "alert": false}`

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## 📄 License
MIT License - see LICENSE file for details