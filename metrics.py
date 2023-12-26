
import numpy as np

# given a list of predictions and a list of labels, return the MSE
def MSE(predictions, labels):
    return np.mean((predictions - labels)**2)

# given a list of predictions and a list of labels, return the MAE
def MAE(predictions, labels):
    return np.mean(np.abs(predictions - labels))

# given a list of predictions and a list of labels, return the RMSE
def RMSE(predictions, labels):
    return np.sqrt(np.mean((predictions - labels)**2))

# given a list of predictions and a list of labels, return the MAPE
def MAPE(predictions, labels):
    return np.mean(np.abs((predictions - labels) / labels)) * 100

# given a list of predictions and a list of labels, return the R2 score
def R2(predictions, labels):
    return 1 - np.sum((predictions - labels)**2) / np.sum((labels - np.mean(labels))**2)



