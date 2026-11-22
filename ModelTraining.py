import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from dataManagement import TrainingData, DESCRIPTIVE_COLS as DESCRIPTIVE, load_params
from models.classificationModels import run_classifications, NOMBRES as NOMBRES_CLF
from models.regressionModels import run_regression, NOMBRES as NOMBRES_REG

DATA_URL = "https://raw.githubusercontent.com/Fabian830348/Bases_Datos/refs/heads/master/Casos_Estudio/Datos/Datos_energia_Caso_Final_ML_2026.csv"
ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
OBJECTIVES = {"classification": "falla_del_equipo", "regression": "consumo_energetico"}

sns.set_theme(style="whitegrid")


def build_structure(objective_type: str, test_size: float, data_route: str = DATA_URL) -> TrainingData:
    ds = TrainingData(DATA_ROUTE=data_route)
    ds.make_split(DESCRIPTIVE=DESCRIPTIVE, OBJECTIVE=OBJECTIVES[objective_type],
                  TEST_SIZE=test_size, objective_type=objective_type)
    ds.scale_data("std")
    return ds


def guardar_figura(fig, nombre):
    fig.savefig(ART / nombre, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_correlaciones(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Mapa de correlaciones", fontweight="bold")
    guardar_figura(fig, "correlaciones.png")


def fig_matrices(res_clf):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, cm) in zip(axes, res_clf["matrices"].items()):
        ConfusionMatrixDisplay(cm, display_labels=["No", "Sí"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(NOMBRES_CLF[key], fontweight="bold")
    fig.tight_layout()
    guardar_figura(fig, "matrices_confusion.png")


def fig_real_vs_pred(ds_reg, res_reg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    y = ds_reg.y_test
    for ax, (key, pred), color in zip(axes, res_reg["predictions"].items(), ["#4f46e5", "#0891b2", "#f59e0b"]):
        ax.scatter(y, pred, s=8, alpha=0.5, color=color)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_title(NOMBRES_REG[key], fontweight="bold")
        ax.set_xlabel("Consumo real (kWh)")
        ax.set_ylabel("Consumo predicho (kWh)")
    fig.tight_layout()
    guardar_figura(fig, "real_vs_predicho.png")


def fig_coeficientes(res_reg):
    c = res_reg["coefs"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(c["Variable"], c["Coeficiente"], color="#4f46e5", edgecolor="black")
    ax.set_title("Peso de cada variable en la regresión lineal", fontweight="bold")
    ax.set_xlabel("kWh por desviación estándar")
    guardar_figura(fig, "coeficientes.png")


def tabla_parametros(res, nombres):
    return pd.DataFrame([{"Modelo": nombres[k], "Hiperparámetros": ", ".join(f"{p}={v}" for p, v in bp.items())}
                         for k, bp in res["best_params"].items()])


def train_all():
    ART.mkdir(exist_ok=True)
    cfg = load_params()
    test_size, optimize = cfg["test_size"], cfg.get("optimize", True)

    ds_clf = build_structure("classification", test_size)
    ds_reg = build_structure("regression", test_size)
    data = ds_clf.data

    res_clf = run_classifications(ds_clf, cfg["classification"], optimize)
    res_reg = run_regression(ds_reg, cfg["regression"], optimize)

    joblib.dump({
        "clf_models": res_clf["models"],
        "reg_models": res_reg["models"],
        "clf_scaler": ds_clf.scaler,
        "reg_scaler": ds_reg.scaler,
    }, ART / "modelos.joblib", compress=3)

    data.to_csv(ART / "datos_limpios.csv", index=False)
    data.head(10).round(2).to_csv(ART / "primeras_filas.csv", index=False)
    data.describe().T.round(2).reset_index(names="variable").to_csv(ART / "descriptivas.csv", index=False)
    res_clf["metrics"].to_csv(ART / "metricas_clasificacion.csv", index=False)
    res_clf["thresholds"].to_csv(ART / "umbrales.csv", index=False)
    res_reg["metrics"].to_csv(ART / "metricas_regresion.csv", index=False)
    res_reg["coefs"].to_csv(ART / "coeficientes.csv", index=False)
    tabla_parametros(res_clf, NOMBRES_CLF).to_csv(ART / "parametros_clasificacion.csv", index=False)
    tabla_parametros(res_reg, NOMBRES_REG).to_csv(ART / "parametros_regresion.csv", index=False)

    with open(ART / "resumen.json", "w", encoding="utf-8") as f:
        json.dump({
            "filas_usadas": len(data),
            "filas_originales": len(ds_clf.raw_data),
            "mejor_clasificacion": res_clf["best"],
            "mejor_regresion": res_reg["best"],
            "test_size": test_size,
            "optimizado": optimize,
            "rangos": data[DESCRIPTIVE].agg(["min", "max"]).to_dict(),
        }, f, ensure_ascii=False, indent=2)

    fig_correlaciones(data)
    fig_matrices(res_clf)
    fig_real_vs_pred(ds_reg, res_reg)
    fig_coeficientes(res_reg)
    return res_clf, res_reg


if __name__ == "__main__":
    res_clf, res_reg = train_all()
    print("\nClasificación\n", res_clf["metrics"].to_string(index=False), "\nMejor:", res_clf["best"])
    print("\nRegresión\n", res_reg["metrics"].to_string(index=False), "\nMejor:", res_reg["best"])
    print(f"\nArchivos guardados en {ART}")
