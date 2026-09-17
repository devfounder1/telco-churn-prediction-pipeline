import onnx
import logging
import onnx.checker
import torch
import torch.nn as nn 

logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s : %(message)s')

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
    
logging.info("Загрузка лучших весов модели ...")

model = TabularMLP()
try:
    model.load_state_dict(torch.load("BestModel_MLP.pth", map_location='cpu', weights_only=True))
    logging.info("Модель успешно загружена !!!")
except Exception as e: 
    logging.error(f"Модель не была загружена, ошибка : {e}!!!")
    raise

model.eval()

dummy_input = torch.randn(1, 44, dtype=torch.float32)
logging.info(f"Dummy_input успешно создан | форма : {dummy_input.shape} | тип : {dummy_input.dtype}")

onnx_file_path = "churn_mlp.onnx"
logging.info("Начинаю экспорт в ONNX")

torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input_features'],
    output_names=['churn_logits'],
    dynamic_axes={
        'input_features' : {0 : 'batch_size'},
        'churn_logits' : {0 : 'batch_size'}
    } 
)

logging.info(f"Модель успешно экспортирована в {onnx_file_path} !!!")

try: 
    onnx_model = onnx.load(onnx_file_path)
    
    # onnx.checker.check_model(onnx_model) — ПРОВЕРЯЕТ модель на корректность.
    # Что проверяется:
    # 1. Все ли узлы графа соединены правильно?
    # 2. Соответствуют ли типы данных (float32, int64 и т.д.)?
    # 3. Правильные ли размерности у тензоров?
    # 4. Нет ли «битых» операций?
    onnx.checker.check_model(onnx_model)

    logging.info("ONNX модель прошла валидацию! Все проверки пройдены")

except Exception as e : 
    logging.error(f"Ошибка валидации ONNX-модели: {e}")
    raise

logging.info("Экспорт модели в ONNX прошел отлично ! Файлы готовы к использованию")