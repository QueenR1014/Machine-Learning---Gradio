import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from processing.cleaning import (RUTA_LIMPIOS, DESCRIPTIVE_COLS, OBJECTIVES,
                                 cargar_crudos, clean_data, guardar_limpios)
from models.dataManagement import TrainingData

TEST_SIZE = 0.2

RESULTADOS = ROOT / "resultados"
DIR_MODELOS = RESULTADOS / "modelos"
DIR_METRICAS = RESULTADOS / "metricas"
DIR_GRAFICAS = RESULTADOS / "graficas"

# Llaves de los diccionarios de run_classifications / run_regressions -> formato en que se guarda cada modelo
MODELOS_CLF = {"LogReg": "joblib", "RandForest": "joblib", "NNC": "keras"}
MODELOS_REG = {"LinearReg": "joblib", "SVR": "joblib", "NNR": "keras"}


def construir_estructura(objective_type: str = "classification", ruta=RUTA_LIMPIOS,
                         test_size: float = TEST_SIZE) -> TrainingData:
    """TrainingData con el split 80/20 (seed 2026, estratificado en clasificación) y StandardScaler."""
    ds = TrainingData(DATA_ROUTE=str(ruta))
    ds.make_split(DESCRIPTIVE=DESCRIPTIVE_COLS, OBJECTIVE=OBJECTIVES[objective_type],
                  TEST_SIZE=test_size, objective_type=objective_type)
    ds.scale_data("std")
    return ds


def train_all_models(ruta=RUTA_LIMPIOS, test_size: float = TEST_SIZE, desde_crudos: bool = False,
                     optimize: bool = True) -> dict:
    """
    Entrena todos los modelos (clasificación y regresión) y exporta los resultados a resultados/.
    - ruta: datos limpios (por defecto data/datos_limpios.csv).
    - desde_crudos=True: primero limpia data/datos_crudos.csv y sobrescribe datos_limpios.csv.
    - optimize=False: parámetros de model_cfg.json.
      optimize=True (por defecto): búsqueda de hiperparámetros.
    Retorna {"clasificacion": tabla de métricas, "regresion": tabla de métricas}.
    """
    if desde_crudos:
        guardar_limpios(clean_data(cargar_crudos()), ruta)

    # Imports aquí: cargan TensorFlow (lento) y así el resto del módulo se puede usar sin él
    from models.classificationModels import run_classifications
    from models.regressionModels import run_regressions

    # ---- Clasificación ----
    ds_clf = construir_estructura("classification", ruta, test_size)
    modelos_clf, predicciones_clf = run_classifications(ds_clf, optimize=optimize)

    metricas_clf = calcular_metricas_clasificacion(ds_clf, predicciones_clf)
    guardar_modelos(modelos_clf, ds_clf.scaler, "clf")
    guardar_metricas(metricas_clf, predicciones_clf, ds_clf, "clasificacion")
    guardar_graficas_clasificacion(ds_clf, predicciones_clf)

    # ---- Regresión ----
    ds_reg = construir_estructura("regression", ruta, test_size)
    modelos_reg, predicciones_reg = run_regressions(ds_reg, optimize=optimize)

    metricas_reg = calcular_metricas_regresion(ds_reg, predicciones_reg)
    guardar_modelos(modelos_reg, ds_reg.scaler, "reg")
    guardar_metricas(metricas_reg, predicciones_reg, ds_reg, "regresion")
    guardar_graficas_regresion(ds_reg, predicciones_reg)

    return {"clasificacion": metricas_clf, "regresion": metricas_reg}


# ---------------- Exportación ----------------

def _crear_carpetas():
    for carpeta in (DIR_MODELOS, DIR_METRICAS, DIR_GRAFICAS):
        carpeta.mkdir(parents=True, exist_ok=True)


def calcular_metricas_clasificacion(ds: TrainingData, predicciones: dict) -> pd.DataFrame:
    """Accuracy, recall, specificity, precision y F1 de cada modelo sobre el mismo conjunto de prueba."""
    from models.classificationModels import classification_metrics, NOMBRES
    filas = [{"Modelo": NOMBRES.get(key, key), **classification_metrics(ds.y_test, pred)}
             for key, pred in predicciones.items()]
    return pd.DataFrame(filas).round(3)


def calcular_metricas_regresion(ds: TrainingData, predicciones: dict) -> pd.DataFrame:
    """R², MSE, RMSE y MAE de cada modelo sobre el mismo conjunto de prueba."""
    from models.regressionModels import regression_metrics, NOMBRES
    filas = [{"Modelo": NOMBRES.get(key, key), **regression_metrics(ds.y_test, pred)}
             for key, pred in predicciones.items()]
    return pd.DataFrame(filas).round(4)


def guardar_modelos(modelos: dict, scaler, sufijo: str):
    _crear_carpetas()
    for key, modelo in modelos.items():
        if isinstance(modelo, tuple):              # Keras: (model, history)
            red, history = modelo
            red.save(DIR_MODELOS / f"{key}.keras")
            pd.DataFrame(history.history).to_csv(DIR_METRICAS / f"historial_{key}.csv", index_label="epoca")
        else:                                      # sklearn
            joblib.dump(modelo, DIR_MODELOS / f"{key}.joblib", compress=3)
    joblib.dump(scaler, DIR_MODELOS / f"scaler_{sufijo}.joblib")


def guardar_metricas(metricas: pd.DataFrame, predicciones: dict, ds: TrainingData, nombre: str):
    _crear_carpetas()
    metricas.to_csv(DIR_METRICAS / f"metricas_{nombre}.csv", index=False)
    # y real vs lo que predijo cada modelo, fila por fila del conjunto de prueba
    tabla = pd.DataFrame({"y_real": ds.y_test.to_numpy(), **predicciones}, index=ds.y_test.index)
    tabla.to_csv(DIR_METRICAS / f"predicciones_{nombre}.csv", index_label="indice")


def guardar_graficas_clasificacion(ds: TrainingData, predicciones: dict):
    from models.classificationModels import NOMBRES
    _crear_carpetas()
    fig, axes = plt.subplots(1, len(predicciones), figsize=(5 * len(predicciones), 4.5))
    for ax, (key, pred) in zip(axes, predicciones.items()):
        ConfusionMatrixDisplay(confusion_matrix(ds.y_test, pred, labels=[0, 1]),
                               display_labels=["No falla", "Falla"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(NOMBRES.get(key, key), fontweight="bold")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    fig.suptitle("Matrices de confusión (conjunto de prueba)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DIR_GRAFICAS / "matrices_confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def guardar_graficas_regresion(ds: TrainingData, predicciones: dict):
    from models.regressionModels import NOMBRES, regression_metrics
    _crear_carpetas()
    y = ds.y_test.to_numpy()
    lim = [y.min() - 5, y.max() + 5]
    colores = ["#4f46e5", "#2a9d8f", "#f4a261"]
    fig, axes = plt.subplots(1, len(predicciones), figsize=(5 * len(predicciones), 4.8))
    for ax, (key, pred), color in zip(axes, predicciones.items(), colores):
        ax.scatter(y, pred, s=10, alpha=0.5, color=color)
        ax.plot(lim, lim, "r--", linewidth=1.5)
        ax.set_title(f"{NOMBRES.get(key, key)}\nR² = {regression_metrics(y, pred)['R2']:.4f}", fontweight="bold")
        ax.set_xlabel("Consumo real (kWh)")
        ax.set_ylabel("Consumo predicho (kWh)")
    fig.suptitle("Consumo real vs predicho (conjunto de prueba)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DIR_GRAFICAS / "real_vs_predicho.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------- Carga (para la app) ----------------

TAREAS = {
    "clasificacion": {"modelos": MODELOS_CLF, "sufijo": "clf", "graficas": ["matrices_confusion"]},
    "regresion": {"modelos": MODELOS_REG, "sufijo": "reg", "graficas": ["real_vs_predicho"]},
}


def _archivos(tarea: str) -> list:
    t = TAREAS[tarea]
    archivos = [DIR_MODELOS / f"{key}.{ext}" for key, ext in t["modelos"].items()]
    archivos += [DIR_MODELOS / f"scaler_{t['sufijo']}.joblib",
                 DIR_METRICAS / f"metricas_{tarea}.csv",
                 DIR_METRICAS / f"predicciones_{tarea}.csv"]
    archivos += [DIR_GRAFICAS / f"{g}.png" for g in t["graficas"]]
    return archivos


def modelos_listos() -> bool:
    """True si están todos los archivos que guarda train_all_models (clasificación y regresión)."""
    return all(f.is_file() for tarea in TAREAS for f in _archivos(tarea))


def cargar_resultados() -> dict:
    """Carga de resultados/ lo necesario para la app, sin volver a entrenar."""
    faltan = [f.name for tarea in TAREAS for f in _archivos(tarea) if not f.is_file()]
    if faltan:
        raise FileNotFoundError(f"No hay modelos entrenados (faltan {faltan}). Corre train_all_models().")

    resultados = {}
    for tarea, t in TAREAS.items():
        modelos = {}
        for key, ext in t["modelos"].items():
            ruta = DIR_MODELOS / f"{key}.{ext}"
            if ext == "keras":
                import keras
                modelos[key] = keras.models.load_model(ruta)   # solo el modelo, sin history
            else:
                modelos[key] = joblib.load(ruta)
        resultados[tarea] = {
            "modelos": modelos,
            "scaler": joblib.load(DIR_MODELOS / f"scaler_{t['sufijo']}.joblib"),
            "metricas": pd.read_csv(DIR_METRICAS / f"metricas_{tarea}.csv"),
            "predicciones": pd.read_csv(DIR_METRICAS / f"predicciones_{tarea}.csv", index_col="indice"),
            "graficas": {g: DIR_GRAFICAS / f"{g}.png" for g in t["graficas"]},
        }
    return resultados


if __name__ == "__main__":
    for tarea, tabla in train_all_models().items():
        print(f"\n{tarea}\n{tabla.to_string(index=False)}")
