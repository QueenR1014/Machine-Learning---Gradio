import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from models.classificationModels import run_classifications
from models.regressionModels import run_regression
from dataManagement import TrainingData, DESCRIPTIVE_COLS as DESCRIPTIVE

DATA_URL = "https://raw.githubusercontent.com/Fabian830348/Bases_Datos/refs/heads/master/Casos_Estudio/Datos/Datos_energia_Caso_Final_ML_2026.csv"
LOCAL_CSV = Path(__file__).parent / "data" / "Datos_energia_Caso_Final_ML_2026.csv"
CACHE_PATH = Path(__file__).parent / "models" / "trained_models.joblib"


def resolve_data_route() -> str:
    if not LOCAL_CSV.is_file():
        LOCAL_CSV.parent.mkdir(exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(LOCAL_CSV, index=False)
    return str(LOCAL_CSV)


DATA_ROUTE = resolve_data_route()
OBJECTIVES = {"classification": "falla_del_equipo", "regression": "consumo_energetico"}
CFG_PATH = Path(__file__).parent / "models" / "model_cfg.json"


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_structure(objective_type: str, test_size: float) -> TrainingData:
    ds = TrainingData(DATA_ROUTE=DATA_ROUTE)
    ds.make_split(DESCRIPTIVE=DESCRIPTIVE, OBJECTIVE=OBJECTIVES[objective_type],
                  TEST_SIZE=test_size, objective_type=objective_type)
    ds.scale_data()
    return ds


def _signature() -> str:
    h = hashlib.md5()
    h.update(CFG_PATH.read_bytes())
    h.update(Path(DATA_ROUTE).read_bytes())
    return h.hexdigest()


def train_all(use_cache: bool = True):
    sig = _signature()
    if use_cache and CACHE_PATH.is_file():
        saved = joblib.load(CACHE_PATH)
        if saved["sig"] == sig:
            return saved["result"]

    result = _train_all()
    joblib.dump({"sig": sig, "result": result}, CACHE_PATH)
    return result


def _train_all():
    cfg = load_cfg()
    test_size = cfg["test_size"]

    ds_clf = build_structure("classification", test_size)
    ds_reg = build_structure("regression", test_size)

    res_clf = run_classifications(ds_clf, cfg["classification"])
    res_reg = run_regression(ds_reg, cfg["regression"])
    return cfg, ds_clf, ds_reg, res_clf, res_reg


if __name__ == "__main__":
    cfg, ds_clf, ds_reg, res_clf, res_reg = train_all()
    print(f"Filas usadas: {len(ds_clf.data)} de {len(ds_clf.raw_data)}\n")
    print("Clasificación\n", res_clf["metrics"], "\nMejor:", res_clf["best"], "\n")
    print("Regresión\n", res_reg["metrics"], "\nMejor:", res_reg["best"])
