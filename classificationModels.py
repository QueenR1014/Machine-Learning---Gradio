import pandas as pd

# Preprocesamiento
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Definición Estructura de datos
from dataManagement import TrainingData

try:
    from interface import DATA_DIRECTORY, OBJECTIVE, DESCRIPTIVE, TEST_SIZE # El módulo de interfaz debe recibir un directorio que el usuario ingresa
training_data = TrainingData(DATA_DIRECTORY)
training_data.make_split(DESCRIPTIVE=DESCRIPTIVE, OBJECTIVE=OBJECTIVE, TEST_SIZE=TEST_SIZE)

def LogClassification(X: pd.DataFrame | pd.Series, y: pd.Series):
    pass