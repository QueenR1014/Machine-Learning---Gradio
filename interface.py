# Módulo place holder para la verdadera apliación con gradio

# Modelos de clasificación
from models.classificationModels import LogClassification
from models.regressionModels import LinReg

# Las siguientes variables deberían ser atrapadas por la aplicación cuando el usuario las ingrese
# 1. La ruta de los datos 
DATA_ROUTE = "https://raw.githubusercontent.com/Fabian830348/Bases_Datos/refs/heads/master/Casos_Estudio/Datos/Datos_energia_Caso_Final_ML_2026.csv"
#2. Las variables objetivo
OBJECTIVE = ["Falla del Equipo"]
#3. Las variables descriptivas
