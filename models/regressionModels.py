import numpy as np

from models.dataManagement import TrainingData, load_params

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR as SVRModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import  GridSearchCV

from keras.models    import Sequential
from keras.layers    import Dense, Input
from tqdm.keras      import TqdmCallback

VALID_LINREG_PARAMS = []
VALID_SVR_PARAMS = ["kernel", "C", "epsilon", "gamma"]
VALID_NNR_PARAMS = []

NOMBRES = {
    "LinearReg": "Regresión lineal",
    "SVR": "SVR",
    "NNR": "Red neuronal",
}


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def predecir_consumo(modelo, X) -> np.ndarray:
    if isinstance(modelo, tuple):
        modelo = modelo[0]
    if hasattr(modelo, "get_params"):
        return np.asarray(modelo.predict(X)).ravel()
    return np.asarray(modelo.predict(np.asarray(X, dtype="float32"), verbose=0)).ravel()


def LinearReg(Data_Structure: TrainingData):
    # PARA ESTE MODELO NO HAY OPTIMIZACIÓN PORQUE RESOLVEMOS POR OLS
    # print(10*"#" + " ENTRENANDO LINEAR REG " + 10*"#")
    if Data_Structure.X_train is None:
        raise RuntimeError("Primero debes llamar a TrainingData.make_split()")

    modelo = LinearRegression()
    modelo.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
    return modelo

def SVR(Data_Structure: TrainingData, optimize: bool = True):
    # print(10*"#" + " ENTRENANDO SVR " + 10*"#")
    if Data_Structure.scaler is None:
        raise RuntimeError("Primero debes llamar a TrainingData.scale_data()")

    if not optimize:
        raw_params = load_params()['regression']['SVR']
        SVR_params = {k:v for k,v in raw_params.items() if k in VALID_SVR_PARAMS}
        modelo = SVRModel(**SVR_params)
        modelo.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return modelo
    elif optimize:
        modelo = SVRModel()
        grid_params = {
            "kernel": ["rbf", "linear", "poly"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.5],
            "gamma": ["scale", "auto", 0.01, 0.1]
        }

        Grid = GridSearchCV(
            modelo, 
            grid_params,
            cv= 10,
            verbose = 1
        )

        Grid.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return Grid.best_estimator_

def NNR(Data_Structure: TrainingData, optimize: bool = True):

    # print(10*"#" + " ENTRENANDO NEURAL NETWORK REGRESSION " + 10*"#")

    if Data_Structure.scaler is None:
        raise RuntimeError(f"Primero debes llamar a TrainingData.scale()")

     # Fijamos semilla para reproductibilidad
    np.random.seed(2)

    ds = Data_Structure
    n_columnas = ds.X_train.shape[1]

    # Crear separación de validación
    if ds.X_val is None:
        ds.make_NN_splits(objective_type="regression")
    
    X_train_s = ds.X_train_scaled
    X_val_s = ds.X_val

    if not optimize:
        NN_params = load_params()['regression']['NeuralNetwork']

        model = Sequential()
        model.add(Input(shape=(n_columnas,)))

        # Inserción de capas ocultas
        layers = NN_params['hidden_layer_sizes']
        for neurons in layers:
            model.add(Dense(neurons, activation=NN_params['activation']))

        # Salida de regresión
        model.add(Dense(1,activation='linear'))

        model.compile(loss='mse',       # regresión
              optimizer='adam',
              metrics=['mae'])  # regresión

        # Entrenar el modelo usando el conjunto de validación explícitamente
        history = model.fit(X_train_s,
                            ds.y_train,
                            epochs=NN_params['max_iter'],
                            validation_data=(X_val_s, ds.y_val),
                            verbose=0,
                            callbacks=[TqdmCallback()] )
        return model, history
    if optimize:

        model = Sequential()
        # número de entradas
        model.add(Input(shape=(n_columnas,)))

        #capas ocultas
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))    # Para regresión

        #modelo de redes neuronales
        # Compilar el modelo
        model.compile(loss='mse',       # regresión
                    optimizer='adam',
                    metrics=['mae'])  # regresión

        # Entrenar el modelo usando el conjunto de validación explícitamente
        history = model.fit(X_train_s,
                            ds.y_train,
                            epochs=50,
                            validation_data=(X_val_s, ds.y_val),
                            verbose=0,
                            callbacks=[TqdmCallback()] )
        return model, history


def run_regressions(Data_Structure: TrainingData, optimize: bool = True):
    # Entrena y predice con cada uno de los modelos de regresión sobre el mismo split.
    # Retorna:
    #   modelos:      {"LinearReg": modelo, "SVR": modelo, "NNR": (model, history)}
    #   predicciones: {"LinearReg": array, "SVR": array, "NNR": array} en kWh sobre ds.X_test_scaled
    ds = Data_Structure

    modelos = {
        "LinearReg": LinearReg(Data_Structure=ds),
        "SVR": SVR(Data_Structure=ds, optimize=optimize),
        "NNR": NNR(Data_Structure=ds, optimize=optimize)
    }

    predicciones = {key: predecir_consumo(modelo, ds.X_test_scaled) for key, modelo in modelos.items()}

    return modelos, predicciones
