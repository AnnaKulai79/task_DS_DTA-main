import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Налаштовуємо підключення до вашої бази даних
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Завантажуємо модель
# Використовуємо назву, яку ми реєстрували: "TelcoChurnModel"
# Та стан "Production" (якщо ви його призначили в UI) або просто версію
model_uri = "models:/TelcoChurnModel/Production" 
model = mlflow.sklearn.load_model(model_uri)

# 3. Ініціалізуємо FastAPI
app = FastAPI(title="Telco Churn Prediction API")

# Описуємо структуру вхідних даних для перевірки (валідації)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "API для прогнозування відтоку клієнтів працює!"}

@app.post("/predict")
def predict(data: CustomerData):
    # Перетворюємо вхідні дані у DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Робимо прогноз імовірності (для XGBoost/ExtraTrees)
    # [0][1] бере імовірність саме позитивного класу (що клієнт піде)
    proba = model.predict_proba(df)[0][1]
    
    return {
        "churn_probability": round(float(proba), 4),
        "risk_level": "High" if proba > 0.4 else "Low"
    }