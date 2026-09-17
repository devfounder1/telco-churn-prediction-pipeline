from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnx
import joblib
import logging 
import torch
import torch.nn as nn 
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')

app = FastAPI(
    title='Telco Customer Churn Prediction',
    description='End-to-end машинное обучение пайплайн для предсказания оттока клиентов телеком-компании.',
    version='1.0.0'
)

class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(44, 64)
        self.Relu = nn.ReLU()
        self.DropOut = nn.Dropout(0.3)
        self.Linear2 = nn.Linear(64, 1)
            
    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.DropOut(x)
        x = self.Linear2(x)
        return x
    
logging.info("Загрука процессора и модели")
try: 
    preprocessor = joblib.load("preprocessor.pkl")
    
    model = TabularMLP()
    
    model.load_state_dict(torch.load("BestModel_MLP.pth", map_location='mps', weights_only=True))
    model.eval()
    
    logging.info("Все артефакты успешно загружены! ")
    
except Exception as e: 
    logging.info("Произошла ошибка загрузки артефактов !")
    raise e

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
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

@app.post("/Predict")
async def predict_churn(customer : CustomerData):
    try: 
        df_new = pd.DataFrame([customer.model_dump()])

        X_processed = preprocessor.transform(df_new)
        
        input_tensor = torch.tensor(X_processed, dtype=torch.float32)
        
        with torch.no_grad():
            logit = model(input_tensor)
            
            churn_probability = torch.sigmoid(logit).item()
        
        prediction_label = "Churn уйдет" if churn_probability > 0.5 else "No churn (останется)"
        
        return{
            "churn_probability" : round(churn_probability, 4),
            "Prediction" : prediction_label,
            "message" : "Предсказание успешно сгенерированно"
        }
        
    except Exception as e:
        logging.error("Ошибка в генерации предсказания!!!")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера : {e}")

@app.get("/health")
async def check_health():
    return{"status" : "ok", "model_loaded" : True}