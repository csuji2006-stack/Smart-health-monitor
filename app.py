import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
from stpyvista import stpyvista
import time
from ml_model import predictor
import requests
from fastapi import FastAPI
import uvicorn
import threading
import streamlit.components.v1 as components

st.set_page_config(page_title="Smart Health Monitor", layout="wide", page_icon="🫀")

# Custom CSS for advanced look
st.markdown("""
<style>
.main {background-color: #0e1117;}
.stApp {background-color: #0e1117;}
h1 {color: #00d4ff; font-family: 'Arial Black';}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              padding: 1rem; border-radius: 10px; color: white;}
.alert-critical {background-color: #ff4757;}
.alert-warning {background-color: #ffa502;}
.alert-normal {background-color: #2ed573;}
</style>
""", unsafe_allow_html=True)

# Sidebar with logo and controls
with st.sidebar:
    st.image("logo.png", width=200)  # 3D Logo [generated_image:11]
    st.title("🩺 Smart Health Monitor")
    hr = st.slider("Heart Rate (bpm)", 50, 200, 75)
    spo2 = st.slider("SpO2 (%)", 80, 100, 98)
    activity = st.slider("Activity Level (%)", 0, 100, 50)
    
    risk_level = predictor.predict_risk(hr, spo2, activity)
    st.metric("Risk Level", risk_level, delta="Stable")
    
    # Hospital URL Button
    if st.button("🏥 AIIMS Hospital Portal", use_container_width=True):
        st.markdown("[Visit National Health Portal](https://www.data.gov.in/catalog/hospital-directory-national-health-portal)")

# 3D Heart Visualization Section
st.header("🔥 3D Interactive Heart Monitor")
plotter = pv.Plotter(window_size=[600, 400])
heart = pv.ParametricEllipsoid(2, 1.5, 1)  # 3D Heart shape
heart['risk'] = np.linspace(0, 1, heart.n_points)
plotter.add_mesh(heart, scalars='risk', cmap='RdYlGn_r', clim=[0,1],
                 show_edges=True, opacity=0.8)
plotter.add_scalar_bar(title="Risk Level")
plotter.camera_position = 'iso'
plotter.background_color = 'black'
stpyvista(plotter, key="heart_3d", height=400) [web:12]

# Real-time Data Stream Dashboard
col1, col2, col3 = st.columns(3)
with col1:
    with st.container():
        st.markdown(f'<div class="metric-card alert-{risk_level.lower()}">'
                   f'<h3>Heart Rate</h3><h1>{hr}</h1></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card alert-{risk_level.lower()}">'
               f'<h3>SpO2</h3><h1>{spo2}%</h1></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card alert-{risk_level.lower()}">'
               f'<h3>Activity</h3><h1>{activity}%</h1></div>', unsafe_allow_html=True)

# Live Stream Simulation
st.subheader("📊 Live Health Stream")
placeholder = st.empty()
for _ in range(50):
    sim_data = predictor.load_data().sample(10)
    sim_data['time'] = pd.date_range(start='now', periods=10, freq='5S')
    fig = px.line(sim_data, x='time', y=['heart_rate', 'spo2'], 
                  title="Real-time Vitals", color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
    placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(0.1)

# Risk Prediction API (FastAPI Backend)
st.subheader("⚙ Risk Prediction API")
api_code = """
@app.post("/predict")
def predict_risk(data: dict):
    hr, spo2, activity = data['heart_rate'], data['spo2'], data['activity_level']
    risk = predictor.predict_risk(hr, spo2, activity)
    return {"risk_level": risk, "alert": True if risk == "Critical" else False}
"""
st.code(api_code, language='python')

if st.button("🚀 Launch API Server"):
    # In production, run: uvicorn main:app --reload
    st.success("API ready at http://localhost:8000/predict")

# GitHub Deployment Instructions
st.subheader("📤 GitHub Deployment")
st.info("""
1. git init && git add .
2. git commit -m "Smart Health Monitor v1.0"
3. git remote add origin YOUR_GITHUB_REPO
4. git push -u origin main
5. Deploy: Streamlit Cloud → New App → GitHub Repo
""")