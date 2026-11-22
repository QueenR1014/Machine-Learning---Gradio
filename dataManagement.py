import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SPLIT_SEED = 2026

RENAME = {
    'Temperatura': 'temperatura',
    'Presión': 'presion',
    'Vibración': 'vibracion',
    'Horas de operación': 'horas_de_operacion',
    'Carga': 'carga',
    'Antigüedad': 'antiguedad',
    'Falla del equipo': 'falla_del_equipo',
    'Consumo energético': 'consumo_energetico',
}

DESCRIPTIVE_COLS = ['temperatura', 'presion', 'vibracion', 'horas_de_operacion', 'carga', 'antiguedad']


class TrainingData():

    def __init__(self, DATA_ROUTE: str):
        self.raw_data = pd.read_csv(DATA_ROUTE)
        self.data = self.clean_data(self.raw_data)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    @staticmethod
    def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
        data = raw.rename(columns=RENAME)
        data['falla_del_equipo'] = data['falla_del_equipo'].astype(str).str.strip().map({'Sí': 1, 'No': 0})
        data = data.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        data['falla_del_equipo'] = data['falla_del_equipo'].astype(int)
        return data

    def make_split(self, DESCRIPTIVE: list, OBJECTIVE: str, TEST_SIZE: float, objective_type: str):
        X = self.data[DESCRIPTIVE]
        y = self.data[OBJECTIVE]
        stratify_param = None if objective_type == "regression" else y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=stratify_param, random_state=SPLIT_SEED, test_size=TEST_SIZE
        )

    def scale_data(self, scaler_type: str = "std"):
        self.scaler = {"std": StandardScaler, "minmax": MinMaxScaler}[scaler_type]()
        cols = self.X_train.columns
        self.X_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=cols)
        self.X_test_scaled = pd.DataFrame(self.scaler.transform(self.X_test), columns=cols)

    def transform_new(self, X_new: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(X_new), columns=X_new.columns)


def load_params(path=None) -> dict:
    import json
    from pathlib import Path
    path = path or Path(__file__).parent / "models" / "model_cfg.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
