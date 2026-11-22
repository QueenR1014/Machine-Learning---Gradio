import pandas as pd

import numpy as np

from models.dataManagement import TrainingData, load_params

from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

from keras.models    import Sequential
from keras.layers    import Dense, Input
from tqdm.keras                 import TqdmCallback

SEED = 2026


# Parámetros válidos para los modelos
VALID_LOGREG_PARAMS = ["C", "solver", "max_iter", "penalty"]
VALID_RANDFOREST_PARAMS = ["n_estimators", "criterion", "max_depth", "min_samples_split","min_samples_leaf","max_features","bootstrap","random_state"]
VALID_NNC_PARAMS = ["hidden_layer_sizes", "activation", "solver", "alpha", "max_iter"]

# Mismas llaves que el diccionario de modelos de run_classifications
NOMBRES = {
    "LogReg": "Regresión logística",
    "RandForest": "Random Forest",
    "NNC": "Red neuronal",
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

def probabilidad_falla(modelo, X) -> np.ndarray:
    """Probabilidad de falla (clase 1) para cualquiera de los 3 modelos.
    Acepta el modelo de sklearn, el modelo de Keras o la tupla (model, history) de NNC."""
    if isinstance(modelo, tuple):
        modelo = modelo[0]
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return np.asarray(modelo.predict(np.asarray(X, dtype="float32"), verbose=0)).ravel()

def threshold_table(model, X_test, y_test, umbrales=(0.5, 0.4, 0.35, 0.3, 0.25)) -> pd.DataFrame:
    prob = probabilidad_falla(model, X_test)
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
        model.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return model

    elif optimize:

        model = LogisticRegression(random_state=SEED)
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
            scoring="f1",   # clases desbalanceadas (75/25): mismo criterio con el que se elige el mejor modelo
            verbose= 1,
            n_jobs=-1
        )

        Grid.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return Grid.best_estimator_

def RandForest(Data_Structure: TrainingData, optimize:bool = True):
    # print(10*"#" + " ENTRENANDO RANDOM FOREST " + 10*"#")

    if Data_Structure.scaler is None:
        raise RuntimeError(f"Primero debes llamar a TrainingData.scale()")
    
    if not optimize:
        raw_params = load_params()['classification']['RandomForest']
        forest_params = {k: v for k, v in raw_params.items() if k in VALID_RANDFOREST_PARAMS}
        # MODELO
        model = RandomForestClassifier(**forest_params)
        model.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return model
    elif optimize:

        model = RandomForestClassifier(random_state=SEED)
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
            scoring="f1",
            verbose=1,
            n_jobs=-1  # Muy recomendado en Random Forest para entrenar árboles en paralelo
        )
        Grid.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return Grid.best_estimator_

def NNC(Data_Structure: TrainingData, optimize:bool = True):
    # print(10*"#" + " ENTRENANDO NEURAL NETWORK CLASSIFICATION " + 10*"#")
    if Data_Structure.scaler is None:
            raise RuntimeError(f"Primero debes llamar a TrainingData.scale()")
    # Fijamos semilla para reproductibilidad
    np.random.seed(2)

    ds = Data_Structure
    n_columnas = ds.X_train.shape[1]

    # Crear separación de validación
    if ds.X_val is None:
        ds.make_NN_splits()
    
    X_train_s = ds.X_train_scaled
    X_val_s = ds.X_val

    if not optimize:
        NN_params = load_params()['classification']['NeuralNetwork']

        model = Sequential()
        model.add(Input(shape=(n_columnas,)))

        # sección de capas ocultas
        layers = NN_params["hidden_layer_sizes"]
        for neurons in layers:
            model.add(Dense(neurons, activation=NN_params["activation"]))

        # capa de salida
        model.add(Dense(1, activation="sigmoid"))

        # compilador / optimizador
        optimizador = "binary_crossentropy"
        model.compile(loss=optimizador,
                      optimizer=NN_params["solver"],
                      metrics = ["accuracy"])
        
        # entrenamiento del modelo
        history = model.fit(
            X_train_s,
            ds.y_train,
            epochs= NN_params["max_iter"],
            validation_data=(X_val_s, ds.y_val),
            verbose=0,
            callbacks=[TqdmCallback()]
        )

        return model, history
    if optimize:
        model = Sequential()
        model.add(Input(shape=(n_columnas,)))

        # Solo usaremos 3 capas ocultas
        model.add(Dense(16,activation="relu"))
        model.add(Dense(8,activation="relu"))
        model.add(Dense(8,activation="relu"))

        # Capa de salida
        model.add(Dense(1, activation="sigmoid"))

        # compilador / optimizador
        optimizador = "binary_crossentropy"
        model.compile(loss=optimizador,
                        optimizer="adam",
                        metrics = ["accuracy"])
        
        history = model.fit(
                    X_train_s,
                    ds.y_train,
                    epochs= 100,
                    validation_data=(X_val_s, ds.y_val),
                    verbose=0,
                    callbacks=[TqdmCallback()]
        )

        return model, history
    
    
def run_classifications(Data_Structure:TrainingData, optimize:bool = True):
    # Entrena y predice con cada uno de los modelos de clasificación sobre el mismo split.
    # Retorna:
    #   modelos:      {"LogReg": modelo, "RandForest": modelo, "NNC": (model, history)}
    #   predicciones: {"LogReg": array 0/1, "RandForest": array 0/1, "NNC": array 0/1} sobre ds.X_test_scaled
    ds = Data_Structure

    modelos = {
        "LogReg": LogClassification(Data_Structure=ds, optimize=optimize),
        "RandForest": RandForest(Data_Structure=ds, optimize=optimize),
        "NNC": NNC(Data_Structure=ds, optimize=optimize) # recordar que son una tupla (model, history)
    }

    predicciones = {}
    for key, modelo in modelos.items():
        if key == "NNC":
            # Keras da probabilidades: umbral 0.5 para pasar a 0/1
            predicciones[key] = (probabilidad_falla(modelo, ds.X_test_scaled) >= 0.5).astype(int)
        else:
            predicciones[key] = np.asarray(modelo.predict(ds.X_test_scaled)).astype(int)

    return modelos, predicciones
