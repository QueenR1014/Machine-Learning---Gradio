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

# Llaves del diccionario de run_classifications -> formato en que se guarda cada modelo
MODELOS_CLF = {"LogReg": "joblib", "RandForest": "joblib", "NNC": "keras"}


def construir_estructura(objective_type: str = "classification", ruta=RUTA_LIMPIOS,
                         test_size: float = TEST_SIZE) -> TrainingData:
    """TrainingData con el split 80/20 (seed 2026, estratificado en clasificación) y StandardScaler."""
    ds = TrainingData(DATA_ROUTE=str(ruta))
    ds.make_split(DESCRIPTIVE=DESCRIPTIVE_COLS, OBJECTIVE=OBJECTIVES[objective_type],
                  TEST_SIZE=test_size, objective_type=objective_type)
    ds.scale_data("std")
    return ds


def train_all_models(ruta=RUTA_LIMPIOS, test_size: float = TEST_SIZE, desde_crudos: bool = False,
                     optimize: bool = True):
    """
    Entrena todos los modelos y exporta los resultados.
    Por defecto usa data/datos_limpios.csv. Con desde_crudos=True primero limpia
    data/datos_crudos.csv y sobrescribe datos_limpios.csv.
    optimize=True usa GridSearchCV (tarda varios minutos); optimize=False usa model_cfg.json.
    Retorna la tabla de métricas de clasificación.
    """
    if desde_crudos:
        guardar_limpios(clean_data(cargar_crudos()), ruta)

    # ---- Clasificación ----
    ds_clf = construir_estructura("classification", ruta, test_size)

    # Import aquí: carga TensorFlow (lento) y así el resto del módulo se puede usar sin él
    from models.classificationModels import run_classifications
    resultado = run_classifications(ds_clf, optimize=optimize)
    if resultado is None:
        raise RuntimeError("run_classifications todavía no retorna (modelos, predicciones).")
    modelos_clf, predicciones_clf = resultado

    metricas_clf = calcular_metricas_clasificacion(ds_clf, predicciones_clf)
    guardar_modelos(modelos_clf, ds_clf.scaler, "clf")
    guardar_metricas(metricas_clf, predicciones_clf, ds_clf, "clasificacion")
    guardar_graficas_clasificacion(ds_clf, predicciones_clf)

    # ---- Regresión (pendiente: modelos de Reina) ----
    # ds_reg = construir_estructura("regression", ruta, test_size)
    # modelos_reg, predicciones_reg = run_regression(ds_reg)

    return metricas_clf


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


# ---------------- Carga (para la app) ----------------

def _archivos_clasificacion():
    archivos = [DIR_MODELOS / f"{key}.{ext}" for key, ext in MODELOS_CLF.items()]
    archivos += [DIR_MODELOS / "scaler_clf.joblib", DIR_METRICAS / "metricas_clasificacion.csv"]
    return archivos


def modelos_listos() -> bool:
    """True si están todos los archivos que guarda train_all_models para clasificación."""
    return all(f.is_file() for f in _archivos_clasificacion())


def cargar_resultados() -> dict:
    """Carga de resultados/ lo necesario para la app, sin volver a entrenar."""
    if not modelos_listos():
        faltan = [f.name for f in _archivos_clasificacion() if not f.is_file()]
        raise FileNotFoundError(f"No hay modelos entrenados (faltan {faltan}). Corre train_all_models().")

    modelos = {}
    for key, ext in MODELOS_CLF.items():
        ruta = DIR_MODELOS / f"{key}.{ext}"
        if ext == "keras":
            import keras
            modelos[key] = keras.models.load_model(ruta)
        else:
            modelos[key] = joblib.load(ruta)

    return {
        "clasificacion": {
            "modelos": modelos,           # NNC se carga solo como modelo (sin history)
            "scaler": joblib.load(DIR_MODELOS / "scaler_clf.joblib"),
            "metricas": pd.read_csv(DIR_METRICAS / "metricas_clasificacion.csv"),
            "predicciones": pd.read_csv(DIR_METRICAS / "predicciones_clasificacion.csv", index_col="indice"),
            "graficas": {"matrices_confusion": DIR_GRAFICAS / "matrices_confusion.png"},
        },
        # "regresion": pendiente (modelos de Reina)
    }


if __name__ == "__main__":
    print(train_all_models())
