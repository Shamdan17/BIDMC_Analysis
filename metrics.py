
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


# given a list of predictions and a list of labels, return a table of metrics
def metrics_table(predictions, labels):
    mse = MSE(predictions, labels)
    mae = MAE(predictions, labels)
    rmse = RMSE(predictions, labels)
    mape = MAPE(predictions, labels)
    r2 = R2(predictions, labels)

    return f'+-------+-------+-------+-------+-------+\n' \
           f'|  MSE  |  MAE  |  RMSE |  MAPE |   R2  |\n' \
           f'+-------+-------+-------+-------+-------+\n' \
           f'| {mse:.3f} | {mae:.3f} | {rmse:.3f} | {mape:.3f} | {r2:.3f} |\n' \
           f'+-------+-------+-------+-------+-------+'




