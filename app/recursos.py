"""
Carga única (al iniciar la app) de datos, modelos pre-entrenados y métricas, y utilidades de predicción.
Usa las funciones del backend sin modificarlas.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from processing.cleaning import (DESCRIPTIVE_COLS, OBJECTIVES, RENAME, RUTA_LIMPIOS,
                                 cargar_crudos, preparar_nuevos)
from training.training import cargar_resultados, modelos_listos

# Nombre bonito de cada variable ("presion" -> "Presión")
ETIQUETAS = {interno: original for original, interno in RENAME.items()}
OBJ_CLF = OBJECTIVES["classification"]
OBJ_REG = OBJECTIVES["regression"]
CLASES = {0: "No falla", 1: "Falla"}

UMBRAL = 0.5


def etiqueta(col: str) -> str:
    return ETIQUETAS.get(col, col)


class Recursos:
    def __init__(self):
        if not modelos_listos():
            raise RuntimeError("No hay modelos entrenados en resultados/. "
                               "Ejecuta primero: python entrenar.py")

        # Importar aquí: cargan TensorFlow
        from models.classificationModels import NOMBRES as NOMBRES_CLF
        from models.regressionModels import NOMBRES as NOMBRES_REG

        self.datos = pd.read_csv(RUTA_LIMPIOS)
        self.crudos = cargar_crudos()
        self.res = cargar_resultados()
        self.nombres_clf = NOMBRES_CLF
        self.nombres_reg = NOMBRES_REG

        # Rango y mediana de cada predictora (para los formularios)
        self.rangos = {c: (float(self.datos[c].min()), float(self.datos[c].median()), float(self.datos[c].max()))
                       for c in DESCRIPTIVE_COLS}

    # ---- accesos cortos ----
    @property
    def clf(self) -> dict:
        return self.res["clasificacion"]

    @property
    def reg(self) -> dict:
        return self.res["regresion"]


def escalar(X: pd.DataFrame, scaler) -> pd.DataFrame:
    """Aplica el scaler guardado conservando nombres de columnas (como X_train_scaled)."""
    Xs = X.astype(float).copy()
    cols = list(getattr(scaler, "feature_names_in_", X.columns))
    Xs[cols] = scaler.transform(X[cols])
    return Xs


def predecir_clasificacion(R: Recursos, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de falla de cada modelo para cada fila de X (columnas = llaves de los modelos)."""
    from models.classificationModels import probabilidad_falla
    Xs = escalar(X, R.clf["scaler"])
    return pd.DataFrame({key: probabilidad_falla(modelo, Xs) for key, modelo in R.clf["modelos"].items()},
                        index=X.index)


def predecir_regresion(R: Recursos, X: pd.DataFrame) -> pd.DataFrame:
    """Consumo energético (kWh) predicho por cada modelo para cada fila de X."""
    from models.regressionModels import predecir_consumo
    Xs = escalar(X, R.reg["scaler"])
    return pd.DataFrame({key: predecir_consumo(modelo, Xs) for key, modelo in R.reg["modelos"].items()},
                        index=X.index)


def registro_desde_inputs(valores) -> pd.DataFrame:
    """Valores de los 6 gr.Number (en orden de DESCRIPTIVE_COLS) -> X validado por preparar_nuevos."""
    faltan = [etiqueta(c) for c, v in zip(DESCRIPTIVE_COLS, valores) if v is None]
    if faltan:
        raise ValueError(f"Completa todos los campos. Faltan: {', '.join(faltan)}")
    X, descartados = preparar_nuevos(list(valores))
    if len(descartados):
        raise ValueError("Hay valores no numéricos en el registro.")
    return X


def leer_lote(ruta) -> tuple:
    """Archivo subido -> (tabla original, X válido, descartados)."""
    from processing.cleaning import _leer_archivo  # misma lectura que usa preparar_nuevos
    original = _leer_archivo(ruta)
    X, descartados = preparar_nuevos(original)
    if X.empty:
        raise ValueError("El archivo no tiene filas válidas con las 6 variables predictoras.")
    return original, X, descartados


def guardar_csv_temporal(df: pd.DataFrame, nombre: str) -> str:
    carpeta = Path(tempfile.gettempdir()) / "ml_gradio"
    carpeta.mkdir(exist_ok=True)
    ruta = carpeta / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    return str(ruta)


def fila_aleatoria_prueba(predicciones: pd.DataFrame, datos: pd.DataFrame, semilla=None) -> pd.Series:
    """Una fila del conjunto de prueba (los índices de predicciones_*.csv son índices de datos_limpios)."""
    rng = np.random.default_rng(semilla)
    idx = rng.choice(predicciones.index.to_numpy())
    return datos.loc[idx]
