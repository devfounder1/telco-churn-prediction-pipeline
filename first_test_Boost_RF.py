import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s')
np.random.seed(42)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

logging.info(f"Размер датасета : {df.shape}")
logging.info(f"Первые пять строк : \n{df.head(5)}")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_total_charges = df['TotalCharges'].isnull().sum()
logging.info(f"Пропуски в TotalCharges: {missing_total_charges} ({missing_total_charges / len(df) * 100:.2f}%)")
logging.info(f"Информация о данных : \n{df.dtypes}")
logging.info(f"Базовая статистика : \n{df.describe()}")

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']

x_train_full, x_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

logging.info(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categ_features = X.select_dtypes(include=['object', 'string']).columns.to_list()
logging.info(f"Найдено категориальных признаков: {len(categ_features)}")
logging.info(f"Список: {categ_features}")

numeric_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categ_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipe, numeric_features),
        ('cat', categ_pipe, categ_features)
    ],
    remainder='drop'
)

x_train_processed = preprocessor.fit_transform(x_train)
x_val_processed = preprocessor.transform(x_val)
x_test_processed = preprocessor.transform(x_test)

logging.info(f"Размер X_train после обработки : {x_train_processed.shape}")
logging.info(f"Размер X_val после обработки : {x_val_processed.shape}")
logging.info(f"Размер X_test после обработки : {x_test_processed.shape}")

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(x_train_processed, y_train)
rf_probs = model_rf.predict_proba(x_test_processed)[:, 1]
logging.info(f"BASELINE | ROC-AUC RandomForest (test): {roc_auc_score(y_test, rf_probs):.4f}")

def objective(trial):
    iterations = trial.suggest_int('iterations', 100, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    depth = trial.suggest_int('depth', 4, 8)

    model_catboost = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        verbose=0,
        random_seed=42
    )

    model_catboost.fit(x_train_processed, y_train)
    cat_prob = model_catboost.predict_proba(x_val_processed)[:, 1]

    return roc_auc_score(y_val, cat_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15, show_progress_bar=True)
logging.info(f"Лучшие параметры : {study.best_params}")
logging.info(f"Лучший ROC-AUC (val): {study.best_value:.4f}")

best_model = CatBoostClassifier(
    iterations=study.best_params['iterations'],
    learning_rate=study.best_params['learning_rate'],
    depth=study.best_params['depth'],
    verbose=0,
    random_seed=42
)
best_model.fit(x_train_processed, y_train)
final_probs = best_model.predict_proba(x_test_processed)[:, 1]
logging.info(f"FINAL | ROC-AUC CatBoost (test): {roc_auc_score(y_test, final_probs):.4f}")