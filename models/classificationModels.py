import numpy as np

from dataManagement import TrainingData, load_params

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
