import torch 
from sklearn.model_selection import train_test_split
import logging 
import pandas as pd
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')
np.random.seed(42)
base_url = url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(base_url)

logging.info(f" первые пять строк датасета : \n {df.head(5)}")
logging.info(f" описание датасета : \n {df.describe()}")
logging.info(f" \n {df.info()}")
logging.info(f" размер датаета : {df.shape}")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'Yes' : 1, 'No' : 0})

X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']

x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.25, random_state=42, stratify=y)
logging.info(f"Размер тренировочной части датасета : {x_train.shape}, Размер тестовой части Датасета : {x_val.shape}")

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
ctaeg_features = X.select_dtypes(include=['object', 'string']).columns.to_list()

numeric_pipe = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categ_pipe = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder())
    ]
)

transforme = ColumnTransformer(
    transformers=[
        ('num', numeric_pipe, numeric_features),
        ('categ', categ_pipe, ctaeg_features)
    ],
    remainder='drop'
)

x_train_transformed = transforme.fit_transform(x_train)
x_val_transformed = transforme.transform(x_val)

logging.info(f"Размер X_train после обработки: {x_train_transformed.shape}")
logging.info(f"Размер X_val после обработки: {x_val_transformed.shape}")

x_train_tensor = torch.tensor(x_train_transformed, dtype=torch.long)
x_test_tensor = torch.tensor(x_val_transformed, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_val, dtype=torch.float32)

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
    
model = TabularMLP()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)