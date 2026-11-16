from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SPLIT_SEED = 2026


# Estructura de datos que se inicializa al cargar los datos
# Mantiene el mismo TrainTestSplit para todos los modelos 
class TrainingData(self):

    def __init__(self, DATA_ROUTE: str, sheet=None):

        # Chequear existencia del directorio
        file_path = Path(DATA_ROUTE)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

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


    # Recibe dos listas en donde una son los nombres de las columnas descriptivas y la otra de las columnas objetivo
    def make_split(self,DESCRIPTIVE: list, OBJECTIVE: list, TEST_SIZE: float):

        X = self.data[DESCRIPTIVE]
        y = self.data[OBJECTIVE]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                stratify=y,
                random_state= SPLIT_SEED,
                test_size= TEST_SIZE
                )


