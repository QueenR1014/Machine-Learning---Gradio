import pandas as pd

from dataManagement import TrainingData
# Preprocesamiento
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




# Devuelve un modelo LogisticRegression entrenado con el training_data.X_train y training_data.y_train
def LogClassification(X: pd.DataFrame | pd.Series, y: pd.Series):
    pass

def run_classifications(Data_Structure:TrainingData, model_parameters=None):
    # función que entrena y predice con cada uno de los modelos de clasificación
    # retorna los modelos entrenados, las predicciones del data split, y las métricas (accuracy, recall, specificity, precision, f1-score)
    
    # escalado de variables

    pass