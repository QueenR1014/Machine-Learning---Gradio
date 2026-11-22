import numpy as np

from dataManagement import TrainingData, load_params

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from keras.models    import Sequential
from keras.layers    import Dense, Input
from tqdm.keras      import TqdmCallback

VALID_LINREG_PARAMS = []
VALID_SVR_PARAMS = []
VALID_NNR_PARAMS = []


def LinearReg(Data_Structure: TrainingData):
    # PARA ESTE MODELO NO HAY OPTIMIZACIÓN PORQUE RESOLVEMOS POR OLS
    
