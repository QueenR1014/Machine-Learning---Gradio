"""Tarjetas con los hiperparámetros de los modelos pre-entrenados que cargó la app."""
from html import escape

import pandas as pd

from training.training import DIR_METRICAS

# Cómo se eligió cada modelo con optimize=True (mismas rejillas que models/*.py)
BUSQUEDA = {
    "LogReg": {"metodo": "GridSearchCV", "combinaciones": 90, "cv": 10, "metrica": "F1",
               "ajustados": {"C", "penalty", "solver", "max_iter"}},
    "RandForest": {"metodo": "GridSearchCV", "combinaciones": 432, "cv": 10, "metrica": "F1",
                   "ajustados": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                                 "max_features", "criterion"}},
    "SVR": {"metodo": "GridSearchCV", "combinaciones": 144, "cv": 10, "metrica": "R²",
            "ajustados": {"kernel", "C", "epsilon", "gamma"}},
}

# Parámetros que se muestran de cada modelo sklearn, en orden, con su etiqueta
MOSTRAR = {
    "LogReg": {"C": "C (inverso de la regularización)", "penalty": "Penalización", "solver": "Solver",
               "max_iter": "Máximo de iteraciones", "random_state": "Semilla"},
    "RandForest": {"n_estimators": "Número de árboles", "max_depth": "Profundidad máxima",
                   "min_samples_split": "Mínimo de muestras para dividir",
                   "min_samples_leaf": "Mínimo de muestras por hoja", "max_features": "Variables por división",
                   "criterion": "Criterio de división", "bootstrap": "Bootstrap", "random_state": "Semilla"},
    "SVR": {"kernel": "Kernel", "C": "C (penalización)", "epsilon": "Epsilon", "gamma": "Gamma"},
    "LinearReg": {"fit_intercept": "Intercepto", "positive": "Coeficientes solo positivos"},
}

NOTAS = {
    "LinearReg": "Mínimos cuadrados ordinarios: solución cerrada, sin hiperparámetros que ajustar.",
}


def _valor(param: str, v) -> str:
    if v is None:
        return "Sin límite" if param == "max_depth" else "Ninguno"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    return str(v)


def _filas_sklearn(key: str, modelo) -> list:
    params = modelo.get_params()
    ajustados = BUSQUEDA.get(key, {}).get("ajustados", set())
    return [(etiqueta, _valor(p, params[p]), p in ajustados)
            for p, etiqueta in MOSTRAR.get(key, {}).items() if p in params]


def _filas_keras(key: str, modelo) -> list:
    densas = [capa.get_config() for capa in modelo.layers]
    ocultas, salida = densas[:-1], densas[-1]
    optimizador = modelo.optimizer.get_config()
    nombre_opt = optimizador["name"].split("_")[0].capitalize()     # "adam_1" -> "Adam"
    ruta_hist = DIR_METRICAS / f"historial_{key}.csv"
    epocas = len(pd.read_csv(ruta_hist)) if ruta_hist.is_file() else "—"
    return [
        ("Capas ocultas", " → ".join(str(c["units"]) for c in ocultas) + " neuronas", False),
        ("Activación oculta", ocultas[0]["activation"] if ocultas else "—", False),
        ("Activación de salida", salida["activation"], False),
        ("Función de pérdida", str(modelo.loss), False),
        ("Optimizador", nombre_opt, False),
        ("Tasa de aprendizaje", f"{float(optimizador['learning_rate']):g}", False),
        ("Épocas", str(epocas), False),
        ("Tamaño de lote", "32 (predeterminado de Keras)", False),
        ("Parámetros entrenables", f"{modelo.count_params():,}", False),
    ]


def _subtitulo(key: str, es_red: bool) -> str:
    if key in BUSQUEDA:
        b = BUSQUEDA[key]
        return (f"{b['metodo']}: {b['combinaciones']} combinaciones, validación cruzada de "
                f"{b['cv']} particiones, se eligió por {b['metrica']}")
    if es_red:
        return "Arquitectura fija definida por el equipo (sin búsqueda de hiperparámetros)"
    return NOTAS.get(key, "")


def tarjetas(modelos: dict, nombres: dict) -> str:
    """HTML con una tarjeta por modelo. Resalta los hiperparámetros elegidos por la búsqueda."""
    html = []
    for key, modelo in modelos.items():
        es_red = not hasattr(modelo, "get_params")
        filas = _filas_keras(key, modelo) if es_red else _filas_sklearn(key, modelo)
        cuerpo = "".join(
            f'<div class="hp-fila{" ajustado" if aj else ""}"><span class="hp-p">{escape(e)}</span>'
            f'<span class="hp-v">{escape(v)}</span></div>' for e, v, aj in filas)
        html.append(f'<div class="hp-card"><div class="hp-titulo">{escape(nombres.get(key, key))}</div>'
                    f'<div class="hp-sub">{escape(_subtitulo(key, es_red))}</div>{cuerpo}</div>')
    leyenda = ('<div class="hp-leyenda"><span class="hp-punto"></span>Valor elegido por la búsqueda de '
               'hiperparámetros; el resto es fijo o el valor por defecto de la librería.</div>')
    return f'<div class="hp-grid">{"".join(html)}</div>{leyenda}'
