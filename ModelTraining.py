# Módulo place holder para la verdadera apliación con gradio

# Modelos de clasificación
from models.classificationModels import run_classifications
from models.regressionModels import run_regression

# Módulo de estructura de datos
from dataManagement import TrainingData

# Las siguientes variables deberían ser atrapadas por la aplicación cuando el usuario las ingrese
# 1. La ruta de los datos 
DATA_ROUTE = "https://raw.githubusercontent.com/Fabian830348/Bases_Datos/refs/heads/master/Casos_Estudio/Datos/Datos_energia_Caso_Final_ML_2026.csv"
#2. Las variables objetivo
# debe lograr discernir si el objetivo es de regresión o clasificación
# Tipo de variable objetivo
objective_type = "classification"
# Columna con la variable objetivo
OBJECTIVE = ["Falla del Equipo"]
#3. Las variables descriptivas
DESCRIPTIVE = []
# 4. El porcentaje de los datos usados para entrenamiento
TEST_SIZE = 0.2
# 5. Parámetros de los modelos
# Estructura JSON con los parámetros de los modelos que escoge el usuario
model_parameters = None

####### CREACIÓN ESTRUCTURA DE DATOS PARA TODOS LOS MODELOS #####
DATA_STRUCTURE = TrainingData(DATA_ROUTE=DATA_ROUTE)
##################### ESCALADO DE VARIABLES #####################
DATA_STRUCTURE.scale_data(scaler_type="")
####################### TRAIN-TEST SPLIT ######################## 
DATA_STRUCTURE.make_split(DESCRIPTIVE=DESCRIPTIVE, OBJECTIVE=OBJECTIVE,TEST_SIZE=TEST_SIZE)

if objective_type == "classification":
    run_classifications()
elif objective_type == "regression":
    run_regression()
else:
    raise ValueError(f"Non valid objective type: {objective_type}")