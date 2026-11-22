import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUTA_CRUDOS = DATA_DIR / "datos_crudos.csv"
RUTA_LIMPIOS = DATA_DIR / "datos_limpios.csv"

# Nombres originales del dataset y nombres internos
RENAME = {
    "Temperatura": "temperatura",
    "Presión": "presion",
    "Vibración": "vibracion",
    "Horas de operación": "horas_de_operacion",
    "Carga": "carga",
    "Antigüedad": "antiguedad",
    "Falla del equipo": "falla_del_equipo",
    "Consumo energético": "consumo_energetico",
}

DESCRIPTIVE_COLS = ["temperatura", "presion", "vibracion", "horas_de_operacion", "carga", "antiguedad"]
OBJECTIVES = {"classification": "falla_del_equipo", "regression": "consumo_energetico"}


def _sin_tildes(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).strip().lower()


def _renombrar(df: pd.DataFrame) -> pd.DataFrame:
    # Acepta los nombres originales ("Presión"), sin tildes ("Presion") o ya internos ("presion")
    mapa = {_sin_tildes(original): interno for original, interno in RENAME.items()}
    return df.rename(columns=lambda c: mapa.get(_sin_tildes(c).replace("_", " "), c))


def _leer_archivo(ruta) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    ext = ruta.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(ruta)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(ruta)
    raise ValueError(f"Formato '{ext}' no soportado. Usa .csv, .xlsx o .xls")


def cargar_crudos(ruta=RUTA_CRUDOS) -> pd.DataFrame:
    """Lee el dataset crudo desde data/datos_crudos.csv."""
    return _leer_archivo(ruta)


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Misma limpieza que el notebook: renombra, Sí/No -> 1/0, todo a numérico y elimina filas con faltantes."""
    data = _renombrar(raw_df)

    faltantes = [c for c in RENAME.values() if c not in data.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en los datos crudos: {faltantes}")

    data = data[list(RENAME.values())].copy()
    data["falla_del_equipo"] = data["falla_del_equipo"].map(_sin_tildes).map({"si": 1, "no": 0})
    data = data.apply(pd.to_numeric, errors="coerce")      # "ML", "PLN", etc. -> NaN
    data = data.dropna().reset_index(drop=True)
    data["falla_del_equipo"] = data["falla_del_equipo"].astype(int)
    return data


def guardar_limpios(df: pd.DataFrame, ruta=RUTA_LIMPIOS) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return ruta


def preparar_nuevos(entrada):
    """
    Prepara datos nuevos para predecir. Acepta:
      - un registro: dict {"temperatura": 55, ...} o lista/tupla de 6 valores en el orden de DESCRIPTIVE_COLS
      - una tabla: DataFrame, o ruta a un .csv / .xlsx / .xls (p. ej. el que entregue el profesor)
    Los nombres de columna pueden venir como en el dataset original ("Presión") o internos ("presion").
    Columnas extra (p. ej. la falla o el consumo) se ignoran.

    Retorna (X, descartados):
      - X: DataFrame con las 6 predictoras, en orden y numéricas. Conserva el índice original
        para poder unir las predicciones con la tabla de entrada.
      - descartados: filas de la entrada que no se pueden usar (faltantes o valores no numéricos).
    """
    if isinstance(entrada, (str, Path)):
        df = _leer_archivo(entrada)
    elif isinstance(entrada, pd.DataFrame):
        df = entrada.copy()
    elif isinstance(entrada, dict):
        df = pd.DataFrame([entrada])
    elif isinstance(entrada, (list, tuple)):
        if len(entrada) != len(DESCRIPTIVE_COLS):
            raise ValueError(f"Se esperaban {len(DESCRIPTIVE_COLS)} valores en este orden: {DESCRIPTIVE_COLS}")
        df = pd.DataFrame([list(entrada)], columns=DESCRIPTIVE_COLS)
    else:
        raise TypeError(f"Tipo de entrada no soportado: {type(entrada).__name__}")

    df = _renombrar(df)
    faltantes = [c for c in DESCRIPTIVE_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas predictoras: {faltantes}")

    # Acepta coma decimal ("2,5"), común en archivos de Excel en español
    X = df[DESCRIPTIVE_COLS].apply(lambda s: pd.to_numeric(
        s if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"))
    invalidas = X.isna().any(axis=1)
    return X[~invalidas], df[invalidas]