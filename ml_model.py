import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import streamlit as st

class HealthRiskPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(random_state=42)
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.is_trained = False
    
    def load_data(self):
        # Simulated + public dataset style data [web:4]
        np.random.seed(42)
        n_samples = 10000
        data = {
            'heart_rate': np.random.normal(75, 15, n_samples),
            'spo2': np.random.normal(98, 2, n_samples),
            'activity_level': np.random.uniform(0, 100, n_samples)
        }
        # Risk labels: 0=Normal, 1=Warning, 2=Critical
        data['risk'] = np.where((data['heart_rate'] > 100) | (data['spo2'] < 92), 2,
                   np.where((data['heart_rate'] > 90) | (data['spo2'] < 95), 1, 0))
        return pd.DataFrame(data)
    
    def train(self):
        df = self.load_data()
        X = df[['heart_rate', 'spo2', 'activity_level']]
        y = df['risk']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.lr_model.fit(X_train_scaled, y_train)
        self.dt_model.fit(X_train_scaled, y_train)
        
        lr_pred = self.lr_model.predict(X_test_scaled)
        st.info(f"LR Accuracy: {classification_report(y_test, lr_pred, output_dict=True)['accuracy']:.2f}")
        self.is_trained = True
    
    def predict_risk(self, hr, spo2, activity):
        if not self.is_trained:
            self.train()
        features = np.array([[hr, spo2, activity]])
        features_scaled = self.scaler.transform(features)
        lr_risk = self.lr_model.predict(features_scaled)[0]
        return ['Normal', 'Warning', 'Critical'][lr_risk]

predictor = HealthRiskPredictor()