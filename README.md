#  Telco Customer Churn Prediction Pipeline

End-to-end машинное обучение пайплайн для предсказания оттока клиентов телеком-компании.

## Задача

Бинарная классификация: предсказать, уйдет ли клиент от оператора связи (`Churn`) на основе его демографических данных, истории использования услуг и платежной информации.

**Метрика:** ROC-AUC (задача с дисбалансом классов)

## Стек технологий

- **Анализ данных:** Pandas, NumPy, Matplotlib, Seaborn
- **ML классика:** Scikit-learn (Pipeline, ColumnTransformer, RandomForest)
- **Градиентный бустинг:** CatBoost + Optuna (байесовская оптимизация гиперпараметров)
- **Deep Learning:** PyTorch (MLP с BCEWithLogitsLoss и pos_weight для дисбаланса)
- **Деплой:** FastAPI + ONNX Runtime

## Структура проекта

pet_project1/
├── first_test.py # Основной скрипт: EDA → Preprocessing → CatBoost + Optuna
├── step_1_eda.py # Исследовательский анализ данных
├── requirements.txt # Зависимости проекта
└── README.md


## Как запустить

