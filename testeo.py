from models.classificationModels import run_classifications
from models.dataManagement import TrainingData, DESCRIPTIVE_COLS, 

DATA_STRUCTURE = TrainingData(DATA_ROUTE="data/datos_limpios.csv")

DATA_STRUCTURE.make_split()
