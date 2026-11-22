import numpy as np

from dataManagement import TrainingData, load_params

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.model_selection import  GridSearchCV

from keras.models    import Sequential
from keras.layers    import Dense, Input
from tqdm.keras      import TqdmCallback

VALID_LINREG_PARAMS = []
VALID_SVR_PARAMS = ["kernel", "C", "epsilon", "gamma"]
VALID_NNR_PARAMS = []


def LinearReg(Data_Structure: TrainingData):
    # PARA ESTE MODELO NO HAY OPTIMIZACIÓN PORQUE RESOLVEMOS POR OLS
    # print(10*"#" + " ENTRENANDO LINEAR REG " + 10*"#")
    if Data_Structure.X_train is None:
        raise RuntimeError("Primero debes llamar a TrainingData.make_split()")

    modelo = LinearRegression()
    modelo.fit(Data_Structure.X_train, Data_Structure.y_train)
    return modelo

def SVR(Data_Structure: TrainingData, optimize: bool = True):
    # print(10*"#" + " ENTRENANDO SVR " + 10*"#")
    if Data_Structure.scaler is None:
        raise RuntimeError("Primero debes llamar a TrainingData.scale_data()")

    if not optimize:
        raw_params = load_params()['regression']['SVR']
        SVR_params = {k:v for k,v in raw_params.items() if k in VALID_SVR_PARAMS}
        modelo = SVR(**SVR_params)
        modelo.fit(Data_Structure.X_train_scaled, Data_Structure.y_train)
        return modelo
    elif optimize:
        modelo = SVR()
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