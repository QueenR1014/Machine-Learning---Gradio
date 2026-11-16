import pandas as pd

# Preprocesamiento
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




# Devuelve un modelo LogisticRegression entrenado con el training_data.X_train y training_data.y_train
def LogClassification(X: pd.DataFrame | pd.Series, y: pd.Series):
    pass