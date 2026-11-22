import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.cleaning import (RUTA_LIMPIOS, DESCRIPTIVE_COLS, OBJECTIVES,
                                 cargar_crudos, clean_data, guardar_limpios)
from models.dataManagement import TrainingData

TEST_SIZE = 0.2

RESULTADOS = ROOT / "resultados"
DIR_MODELOS = RESULTADOS / "modelos"
DIR_METRICAS = RESULTADOS / "metricas"
DIR_GRAFICAS = RESULTADOS / "graficas"


def construir_estructura(objective_type: str = "classification", ruta=RUTA_LIMPIOS,
                         test_size: float = TEST_SIZE) -> TrainingData:
    """TrainingData con el split 80/20 (seed 2026, estratificado en clasificación) y StandardScaler."""
    ds = TrainingData(DATA_ROUTE=str(ruta))
    ds.make_split(DESCRIPTIVE=DESCRIPTIVE_COLS, OBJECTIVE=OBJECTIVES[objective_type],
                  TEST_SIZE=test_size, objective_type=objective_type)
    ds.scale_data("std")
    return ds


def train_all_models(ruta=RUTA_LIMPIOS, test_size: float = TEST_SIZE, desde_crudos: bool = False):
    """
    Entrena todos los modelos y exporta los resultados.
    Por defecto usa data/datos_limpios.csv. Con desde_crudos=True primero limpia
    data/datos_crudos.csv y sobrescribe datos_limpios.csv.
    """
    if desde_crudos:
        guardar_limpios(clean_data(cargar_crudos()), ruta)

    # ---- Clasificación ----
    ds_clf = construir_estructura("classification", ruta, test_size)

    # Import aquí: carga TensorFlow (lento) y así el resto del módulo se puede usar sin él
    from models.classificationModels import run_classifications
    resultado = run_classifications(ds_clf)
    if resultado is None:
        raise RuntimeError("run_classifications todavía no retorna (modelos, predicciones).")
    modelos_clf, predicciones_clf = resultado

    metricas_clf = calcular_metricas_clasificacion(ds_clf, predicciones_clf)
    guardar_modelos(modelos_clf, ds_clf.scaler, "clf")
    guardar_metricas(metricas_clf, predicciones_clf, "clasificacion")
    guardar_graficas_clasificacion(ds_clf, predicciones_clf)

    # ---- Regresión (pendiente: modelos de Reina) ----
    # ds_reg = construir_estructura("regression", ruta, test_size)
    # modelos_reg, predicciones_reg = run_regression(ds_reg)

    return metricas_clf


# ---------------- Pendientes ----------------

def calcular_metricas_clasificacion(ds: TrainingData, predicciones: dict):
    # TODO: tabla con classification_metrics(y_test, pred) por modelo + matrices de confusión.
    # Ojo: NNC predice sobre ds.X_test_NN, así que se compara contra ds.y_test_NN, no ds.y_test.
    return None


def guardar_modelos(modelos: dict, scaler, sufijo: str):
    # TODO: LogReg y RandForest con joblib -> DIR_MODELOS / f"{key}.joblib"
    #       NNC es (model, history): model.save(DIR_MODELOS / "NNC.keras")
    #       scaler con joblib -> DIR_MODELOS / f"scaler_{sufijo}.joblib"
    pass


def guardar_metricas(metricas, predicciones: dict, nombre: str):
    # TODO: CSVs en DIR_METRICAS
    pass


def guardar_graficas_clasificacion(ds: TrainingData, predicciones: dict):
    # TODO: PNG de las matrices de confusión en DIR_GRAFICAS
    pass


def modelos_listos() -> bool:
    # TODO: revisar que existan todos los archivos que guarda guardar_modelos
    return DIR_MODELOS.is_dir() and any(DIR_MODELOS.iterdir())


def cargar_resultados():
    # TODO: cargar modelos, scalers, métricas y gráficas de resultados/ para la app
    raise NotImplementedError


if __name__ == "__main__":
    print(train_all_models())
