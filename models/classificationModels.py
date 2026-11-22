import pandas as pd

import numpy as np

from dataManagement import TrainingData, load_params

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

NOMBRES = {
    "LogReg": "Regresión logística",
    "RandomForest": "Random Forest",
    "NeuralNetwork": "Red neuronal",
}


def build_models(params: dict) -> dict:
    nn = dict(params["NeuralNetwork"])
    nn["hidden_layer_sizes"] = tuple(nn["hidden_layer_sizes"])
    return {
        "LogReg": LogisticRegression(**params["LogReg"]),
        "RandomForest": RandomForestClassifier(**params["RandomForest"]),
        "NeuralNetwork": MLPClassifier(**nn),
    }


def classification_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

def threshold_table(model, X_test, y_test, umbrales=(0.5, 0.4, 0.35, 0.3, 0.25)) -> pd.DataFrame:
    prob = model.predict_proba(X_test)[:, 1]
    filas = [{"Umbral": u, **classification_metrics(y_test, (prob >= u).astype(int))} for u in umbrales]
    return pd.DataFrame(filas).round(3)

from sklearn.model_selection import GridSearchCV

# Parámetros válidos para los modelos
VALID_LOGREG_PARAMS = ["C", "solver", "max_iter", "penalty"]
VALID_RANDFOREST_PARAMS = ["n_estimators", "criterion", "max_depth", "min_samples_split","min_samples_leaf","max_features","bootstrap","random_state"]
VALID_NNC_PARAMS = []

# Devuelve un modelo LogisticRegression entrenado con el training_data.X_train y training_data.y_train
def LogClassification(Data_Structure: TrainingData, optimize:bool=True):
    
    if Data_Structure.scaler is None:
        raise RuntimeError(f"Primero debes llamar a TrainingData.scale()")

    if not optimize:
        # parámetros seleccionados para la regresión logística
        raw_params = load_params()['classification']['LogReg']
        log_params = {k: v for k, v in raw_params.items() if k in VALID_LOGREG_PARAMS}
        # modelo
        model = LogisticRegression(**log_params)
        model.fit(Data_Structure.X_train, Data_Structure.y_train)
        return model

    elif optimize:

        model = LogisticRegression()
        # Lista de diccionarios para evitar combinaciones incompatibles (ej. lbfgs con l1)
        grid_params = [
            {
                "solver": ["lbfgs"],
                "penalty": ["l2"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "max_iter": [100, 500, 1000]
            },
            {
                "solver": ["liblinear", "saga"],
                "penalty": ["l1", "l2"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "max_iter": [100, 500, 1000]
            }
        ]

        # print(10*"#" + " ENTRENANDO LOGREG CON GRID SEARCH" + 10*"#")
        # print(grid_params)
        Grid = GridSearchCV(
            model, 
            grid_params,
            cv= 10,
            verbose= 1
        )

        Grid.fit(Data_Structure.X_train, Data_Structure.y_train)
        return Grid.best_estimator_

def RandForest(Data_Structure: TrainingData, optimize:bool = True):

    if not optimize:
        raw_params = load_params()['classification']['RandomForest']
        forest_params = {k: v for k, v in raw_params.items() if k in VALID_RANDFOREST_PARAMS}
        # MODELO
        model = RandomForestClassifier(**forest_params)
        model.fit(Data_Structure.X_train, Data_Structure.y_train)
    elif optimize:

        model = RandomForestClassifier()
        grid_params = {
            "n_estimators": [100, 200, 300],        # Número de árboles en el bosque
            "max_depth": [None, 10, 20, 30],        # Profundidad máxima de cada árbol (None = sin límite)
            "min_samples_split": [2, 5, 10],        # Mínimo de muestras para dividir un nodo interno
            "min_samples_leaf": [1, 2, 4],          # Mínimo de muestras requeridas en un nodo hoja
            "max_features": ["sqrt", "log2"],       # Número de variables a considerar en cada división
            "criterion": ["gini", "entropy"]        # Función para medir la calidad de la división
        }
        Grid = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=10,
            verbose=1,
            n_jobs=-1  # Muy recomendado en Random Forest para entrenar árboles en paralelo
        )
        return Grid.best_estimator_

def NNC(Data_Structure: TrainingData):
    pass

def run_classifications(Data_Structure:TrainingData):
    # función que entrena y predice con cada uno de los modelos de clasificación
    # retorna los modelos entrenados, las predicciones del data split, y las métricas (accuracy, recall, specificity, precision, f1-score)
    
    # escalado de variables

    pass
