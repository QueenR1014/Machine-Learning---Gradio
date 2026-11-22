import pandas as pd

import numpy as np

from dataManagement import TrainingData, load_params

from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


import tensorflow as tf
from keras.models    import Sequential
from keras.layers    import Dense, Input
from keras.layers    import Dropout
from keras.callbacks            import EarlyStopping, LearningRateScheduler
from tqdm.keras                 import TqdmCallback


# Parámetros válidos para los modelos
VALID_LOGREG_PARAMS = ["C", "solver", "max_iter", "penalty"]
VALID_RANDFOREST_PARAMS = ["n_estimators", "criterion", "max_depth", "min_samples_split","min_samples_leaf","max_features","bootstrap","random_state"]
VALID_NNC_PARAMS = ["hidden_layer_sizes", "activation", "solver", "alpha", "max_iter"]

NOMBRES = {
    "LogReg": "Regresión logística",
    "RandomForest": "Random Forest",
    "NeuralNetwork": "Red neuronal",
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


# Devuelve un modelo LogisticRegression entrenado con el training_data.X_train y training_data.y_train
def LogClassification(Data_Structure: TrainingData, optimize:bool=True):
    # print(10*"#" + " ENTRENANDO LOGREG CON GRID SEARCH" + 10*"#")
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
    # print(10*"#" + " ENTRENANDO RANDOM FOREST " + 10*"#")
   
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
        # print(grid_params)
        Grid = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=10,
            verbose=1,
            n_jobs=-1  # Muy recomendado en Random Forest para entrenar árboles en paralelo
        )
        return Grid.best_estimator_

def NNC(Data_Structure: TrainingData, optimize:bool = True):
    print("En")
    # Fijamos semilla para reproductibilidad
    np.random.seed(2)

    ds = Data_Structure
    n_columnas = ds.X_train.shape[1]

    # Crear separación de validación
    ds.make_NN_splits()
    if ds.scaler is None:
        ds.scale_data(scaler_type="std")
    
    escalar = StandardScaler()

    X_train = ds.X_train
    X_val_s = escalar.transform(ds.X_val)
    X_test_s = escalar.transform(ds.X_test_NN)

    if not optimize:
        NN_params = load_params()['classification']['NeuralNetwork']

        model = Sequential()
        model.add(Input(shape=n_columnas,))

        # sección de capas ocultas
        layers = NN_params["hidden_layer_sizes"]
        for i in range(len(layers)):
            model.add(Dense(layers[i], activation=NN_params["activation"]))

        # capa de salida
        model.add(Dense(1, activation="sigmoid"))

        # compilador / optimizador
        optimizador = "binary_crossentropy" if ds.y.shape[1] == 1 else "categorical_crossentropy" 
        model.compile(loss=optimizador,
                      optimizer=NN_params["solver"],
                      metrics = ["acuracy"])
        
        # entrenamiento del modelo
        history = model.fit(
            X_train,
            ds.y_train,
            epochs= NN_params["max_iter"],
            validation_data=(X_val_s, ds.y_val),
            verbose=0,
            callbacks=[TqdmCallback()]
        )

        return model, history
    if optimize:
        model = Sequential()
        model.add(Input(shape=n_columnas,))

        # Solo usaremos 3 capas ocultas

def build_models(params: dict) -> dict:
    nn = dict(params["NeuralNetwork"])
    nn["hidden_layer_sizes"] = tuple(nn["hidden_layer_sizes"])
    return {
        "LogReg": LogisticRegression(**params["LogReg"]),
        "RandomForest": RandomForestClassifier(**params["RandomForest"]),
        "NeuralNetwork": MLPClassifier(**nn),
    }

def run_classifications(Data_Structure:TrainingData):
    # función que entrena y predice con cada uno de los modelos de clasificación
    # retorna los modelos entrenados, las predicciones del data split, y las métricas (accuracy, recall, specificity, precision, f1-score)
    
    # escalado de variables

    pass
