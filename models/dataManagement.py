from pathlib import Path

import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder

SPLIT_SEED = 2026

# Estructura de datos que se inicializa al cargar los datos
# Mantiene el mismo TrainTestSplit para todos los modelos 
class TrainingData():

    def __init__(self, DATA_ROUTE: str, sheet=None):

        # Chequear existencia del directorio
        file_path = Path(DATA_ROUTE)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Lectura del archivo (solo admite xlsx o csv)
        ext = file_path.suffix.lower()
        if ext == ".csv":
            self.data = pd.read_csv(DATA_ROUTE)   
        elif ext in {".xlsx", ".xls"}:
            self.data = pd.read_excel(DATA_ROUTE, sheet_name= sheet)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .xlsx.")        

        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 

        # Tipo de columnas
        self.numeric_cols = []
        self.categorical_cols = []

        # Atributos de escalado
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        # Atributos de categorización
        self.enconder = None
        self.label_encoder = None

        # Atributos de Redes Neuronales
        self.X_val = None
        self.y_val = None
        self.X_test_NN = None
        self.y_test_NN = None

    # Recibe dos listas en donde una son los nombres de las columnas descriptivas y la otra de las columnas objetivo
    def make_split(
    self, DESCRIPTIVE: list, OBJECTIVE: list, TEST_SIZE: float, objective_type: str):
        X = self.data[DESCRIPTIVE]
        y = self.data[OBJECTIVE]

        # Asigna None cuando el objetivo es regresión
        stratify_param = None if objective_type.strip().lower() == "regression" else y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=stratify_param, random_state=SPLIT_SEED, test_size=TEST_SIZE
        )

    def make_NN_splits(self, val_ratio: float = 0.5, objective_type: str = "classification"):
        # Validación de seguridad para asegurar que make_split() se ejecutó antes
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Primero debes llamar a make_split() antes de crear los splits de la red neuronal.")

        # Mantener la estratificación si es clasificación
        stratify_param = None if objective_type.strip().lower() == "regression" else self.y_test

        self.X_val, self.X_test_NN, self.y_val, self.y_test_NN = train_test_split(
            self.X_test,
            self.y_test,
            stratify=stratify_param,
            random_state=SPLIT_SEED,
            test_size=(1 - val_ratio)  # 0.5 deja 50% para validación y 50% para test_NN
        )

    # Escala las variables numéricas de X_train y X_test.
    # type: "std" (StandardScaler) o "minmax" (MinMaxScaler)
    def scale_data(self, scaler_type: str, exclude_binary: bool = True):

        if self.X_train is None:
            raise RuntimeError("Primero debes llamar a make_split().")

        scalers = {"std": StandardScaler, "minmax": MinMaxScaler}
        if scaler_type not in scalers:
            raise ValueError(f"Scaler '{scaler_type}' no soportado. Usa 'std' o 'minmax'.")

        # Detectar columnas numéricas (bool y texto quedan por fuera)
        num_cols = self.X_train.select_dtypes(include="number").columns.tolist()

        # Opcional: no escalar variables dummy (0/1)
        if exclude_binary:
            num_cols = [c for c in num_cols if self.X_train[c].nunique() > 2]

        self.numeric_cols = num_cols
        self.scaler = scalers[scaler_type]()

        # Copias en float para evitar conflictos de dtype al asignar
        self.X_train_scaled = self.X_train.copy().astype({c: float for c in num_cols})
        self.X_test_scaled = self.X_test.copy().astype({c: float for c in num_cols})

        if num_cols:
            # fit solo con train, transform en ambos
            self.X_train_scaled[num_cols] = self.scaler.fit_transform(self.X_train[num_cols])
            self.X_test_scaled[num_cols] = self.scaler.transform(self.X_test[num_cols])

    def cathegorize_data(self):
        pass

def load_params():
    with open("model_cfg.json", "r", encoding="utf-8") as file:
        return json.load(file)
