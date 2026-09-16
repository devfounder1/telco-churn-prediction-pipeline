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
from sklearn.metrics import roc_auc_score

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
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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

x_train_tensor = torch.tensor(x_train_transformed, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val_transformed, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

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

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
logging.info(f"Вес положительного класса (pos_weight): {pos_weight}")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

# Переводим в тензоры 
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Делаем Loaderы 
train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=64,
)

val_loader = DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=64    
)

EPOCHS = 30

for epoch in range(EPOCHS):
    train_loss = 0.0
    model.train()
    for batch_x, batch_y in train_loader:
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0.0
    
    all_val_probs = []
    all_val_targets = []
    
    with torch.no_grad():
        for batch_x_val, batch_y_val in val_loader:
            outputs_val = model(batch_x_val)
            loss_v = criterion(outputs_val, batch_y_val)
            
            val_loss += loss_v.item()
            
            probs = torch.sigmoid(outputs_val).cpu().numpy()
            all_val_probs.extend(probs)
            all_val_targets.extend(batch_y_val.cpu().numpy())
            
        avg_val_loss = val_loss / len(val_loader)
        val_roc_auc = roc_auc_score(all_val_targets, all_val_probs)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Эпоха: {epoch+1}/{EPOCHS} | Avg_train_loss : {avg_train_loss:.3f} | Avg_val_loss : {avg_val_loss:.3f} | Val ROC-AUC score : {val_roc_auc:.3f} ")


torch.save(model.state_dict(), "BestModel_MLP.pth")