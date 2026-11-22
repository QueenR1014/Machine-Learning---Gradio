import numpy as np
import pandas as pd

from dataManagement import TrainingData

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

NOMBRES = {
    "LinearRegression": "Regresión lineal",
    "SVR": "SVR",
    "NeuralNetwork": "Red neuronal",
}


def build_models(params: dict) -> dict:
    nn = dict(params["NeuralNetwork"])
    nn["hidden_layer_sizes"] = tuple(nn["hidden_layer_sizes"])
    return {
        "LinearRegression": LinearRegression(**params["LinearRegression"]),
        "SVR": SVR(**params["SVR"]),
        "NeuralNetwork": MLPRegressor(**nn),
    }


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def run_regression(Data_Structure: TrainingData, model_parameters: dict):
    ds = Data_Structure
    models = build_models(model_parameters)

    predictions, filas = {}, []
    for key, model in models.items():
        model.fit(ds.X_train_scaled, ds.y_train)
        pred = model.predict(ds.X_test_scaled)
        predictions[key] = pred
        filas.append({"Modelo": NOMBRES[key], **regression_metrics(ds.y_test, pred)})

    metrics = pd.DataFrame(filas).round(3)
    best = metrics.sort_values("R2", ascending=False).iloc[0]["Modelo"]

    base = DummyRegressor(strategy="mean").fit(ds.X_train_scaled, ds.y_train)
    baseline = pd.DataFrame([{"Modelo": "Línea base (media)",
                              **regression_metrics(ds.y_test, base.predict(ds.X_test_scaled))}]).round(3)

    lin = models["LinearRegression"]
    coefs = pd.DataFrame({"Variable": ds.X_train.columns, "Coeficiente (estandarizado)": lin.coef_})
    coefs = coefs.sort_values("Coeficiente (estandarizado)", key=abs, ascending=False).round(2)
    return {"models": models, "predictions": predictions, "metrics": metrics, "best": best,
            "baseline": baseline, "coefs": coefs}
