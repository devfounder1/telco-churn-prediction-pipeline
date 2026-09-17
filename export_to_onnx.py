import onnx
import logging
import onnx.checker
import torch
import torch.nn as nn 

logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s : %(message)s')

class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(44,64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        self.linear2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x
    
logging.info("Загрузка лучших весов модели ...")

model = TabularMLP()
try:
    model.load_state_dict(torch.load("BestModel_MLP.pth", map_location='cpu', weights_only=True))
    logging.info("Модель успешно загружена !!!")
except Exception as e: 
    logging.error("Модель не была загружена, ошибка !!!")
    raise e

model.eval()
dummy_input = torch.randn(1, 44, dtype=torch.float32)



