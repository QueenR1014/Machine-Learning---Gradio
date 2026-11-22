import pandas as pd

from dataManagement import TrainingData

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


def run_classifications(Data_Structure: TrainingData, model_parameters: dict):
    ds = Data_Structure
    models = build_models(model_parameters)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    matrices, filas = {}, []
    for key, model in models.items():
        cv = cross_val_score(make_pipeline(StandardScaler(), clone(model)),
                             ds.X_train, ds.y_train, cv=skf, scoring="f1")
        model.fit(ds.X_train_scaled, ds.y_train)
        pred = model.predict(ds.X_test_scaled)
        matrices[key] = confusion_matrix(ds.y_test, pred, labels=[0, 1])
        filas.append({"Modelo": NOMBRES[key], **classification_metrics(ds.y_test, pred),
                      "F1 (CV media)": cv.mean(), "F1 (CV desv.)": cv.std()})

    metrics = pd.DataFrame(filas).round(3)
    best = metrics.sort_values("F1 (CV media)", ascending=False).iloc[0]["Modelo"]
    return {"models": models, "matrices": matrices, "metrics": metrics, "best": best}


def threshold_table(model, X_test, y_test, umbrales=(0.5, 0.4, 0.35, 0.3, 0.25)) -> pd.DataFrame:
    prob = model.predict_proba(X_test)[:, 1]
    filas = [{"Umbral": u, **classification_metrics(y_test, (prob >= u).astype(int))} for u in umbrales]
    return pd.DataFrame(filas).round(3)
