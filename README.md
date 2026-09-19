# Telco Customer Churn Prediction Pipeline

**End-to-end ML пайплайн для предсказания оттока клиентов телеком-компании с production-ready деплоем**

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Задача

Бинарная классификация: предсказать, уйдёт ли клиент от оператора связи (`Churn`) на основе его демографических данных, истории использования услуг и платёжной информации.

**Метрика:** ROC-AUC (задача с дисбалансом классов ~27% churn rate)

## Возможности

- **Полный EDA** с визуализацией и анализом скрытых пропусков
- **Robust предобработка** с защитой от data leakage (`ColumnTransformer`)
- **Классические модели**: Random Forest, CatBoost + Optuna
- **Deep Learning**: PyTorch MLP с Early Stopping и `pos_weight`
- **Экспорт в ONNX** для кросс-платформенного инференса
- **Docker-контейнеризация** для production-деплоя
- **Красивый веб-интерфейс** на FastAPI + HTML/CSS/JS
- **REST API** с Swagger-документацией

## 📁 Структура проекта

```text
pet_project1/
├── app/
│   └── templates/
│       └── index.html              # Премиальный Dark Mode UI для инференса
├── models/
│   ├── BestModel_MLP.pth           # Веса лучшей PyTorch модели (Early Stopping)
│   ├── preprocessor.pkl            # Scikit-learn ColumnTransformer для новых данных
│   └── churn_mlp.onnx              # Экспортированная модель для кросс-платформенности
├── src/
│   ├── first_test_Boost_RF.py      # Пайплайн: CatBoost + Optuna (Hyperparameter Tuning)
│   ├── second_test_MLP.py          # Пайплайн: PyTorch MLP + pos_weight
│   └── export_to_onnx.py           # Скрипт конвертации .pth в .onnx
├── api.py                          # Точка входа FastAPI приложения
├── Dockerfile                      # Инструкция для сборки production-контейнера
├── requirements_api.txt            # Минимальные зависимости для деплоя
└── README.md                       # Это документация
```

## Результаты моделей

| Модель | ROC-AUC (val) | ROC-AUC (test) | Особенности |
|--------|---------------|----------------|-------------|
| **Random Forest** | - | 0.8158 | Baseline, 100 деревьев |
| **CatBoost + Optuna** | 0.8465 | ~0.846 | 15 trials, лучшие гиперпараметры |
| **PyTorch MLP** | **0.8479** | - | Early Stopping (epoch 22), pos_weight=2.77 |

**Лучшие гиперпараметры CatBoost:**
- `iterations`: 402
- `learning_rate`: 0.0258
- `depth`: 4

**Архитектура MLP:**
- `Linear(44, 64)` → `ReLU` → `Dropout(0.3)` → `Linear(64, 1)`
- Optimizer: AdamW (lr=1e-3)
- Early Stopping: patience=15

## Быстрый старт

### Вариант 1: Локальный запуск (для разработки)

```bash
# 1. Клонируйте репозиторий
git clone <your-repo-url>
cd pet_project1

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Установите зависимости
pip install -r requirements_api.txt

# 4. Запустите API
uvicorn api:app --reload --host 0.0.0.0 --port 8000
Откройте http://127.0.0.1:8000 для веб-интерфейса или http://127.0.0.1:8000/docs для Swagger UI.
```

### Вариант 2: Docker (для production)

```bash
# 1. Соберите образ
docker build -t churn-api .

# 2. Запустите контейнер
docker run -p 8000:8000 churn-api

Откройте http://127.0.0.1:8000 в браузере.
```

## API Документация

### `POST /predict`
Принимает данные клиента и возвращает вероятность оттока.

**Request Headers:**
```http
Content-Type: application/json
```

**Request Body:**
<details>
<summary><b>Нажми, чтобы развернуть полный пример JSON (19 признаков)</b></summary>

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 75.5
}
```
</details>

**Response (`200 OK`):**
```json
{
  "churn_probability": 0.8663,
  "prediction": "Churn (уйдет)",
  "message": "Предсказание успешно сгенерировано"
}
```

---

### `GET /health`
Эндпоинт для проверки работоспособности сервиса (Liveness Probe).

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Тестирование

### Тест с "рискованным" клиентом
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "tenure": 1,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 75.5,
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "PaperlessBilling": "Yes"
  }'

## Технологии
### Core ML
- **Scikit-learn:** ColumnTransformer, Pipeline, OneHotEncoder, StandardScaler
- **CatBoost:** Градиентный бустинг с обработкой категориальных признаков
- **Optuna:** Байесовская оптимизация гиперпараметров
- **PyTorch:** Нейросети с BCEWithLogitsLoss, pos_weight, Early Stopping
### Деплой
- **FastAPI:** Асинхронный веб-фреймворк
- **Uvicorn:** ASGI сервер
- **Pydantic:** Валидация данных
- **ONNX:** Универсальный формат моделей
- **Docker:** Контейнеризация
### Frontend
- **HTML5/CSS3:** Современный Dark Mode UI
- **JavaScript:** Асинхронные запросы к API
- **Jinja2:** Шаблонизация (опционально)

## Ключевые инженерные решения
- **Защита от data leakage:** ColumnTransformer обучается только на train, применяется к val/test через transform()
- **Обработка скрытых пропусков:** TotalCharges содержит строки " " вместо чисел — конвертируется через pd.to_numeric(errors='coerce')
- **Дисбаланс классов:** pos_weight=2.77 в BCEWithLogitsLoss компенсирует меньшинство churn-клиентов
- **Early Stopping:** Автоматическая остановка обучения при отсутствии улучшения (patience=15)
- **CPU-версия PyTorch:** Оптимизация Docker-образа (250 МБ вместо 2.5 ГБ)

## Roadmap
- **Добавить логирование предсказаний в SQLite**
- **Написать unit-тесты для API (pytest)**
- **Интегрировать GitHub Actions для CI/CD**
- **Добавить мониторинг дрейфа данных (data drift)**
- **Реализовать A/B тестирование моделей**
- **Деплой на облачный сервер (AWS/GCP)**
## Датасет
**Telco Customer Churn — 7043 клиента, 21 признак, целевая переменная Churn.**
## Лицензия
**MIT License. См. файл LICENSE для деталей.**
## Автор
Mihail — ML Engineer

⭐ Если проект был полезен — поставьте звезду!