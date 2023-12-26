import numpy as np


# numpify decorator
def numpify(func):
    def wrapper(*args):
        return func(*[np.asarray(x) for x in args])

    return wrapper


# given a list of predictions and a list of labels, return the MSE
@numpify
def MSE(predictions, labels):
    return np.mean((predictions - labels) ** 2)


# given a list of predictions and a list of labels, return the MAE
@numpify
def MAE(predictions, labels):
    return np.mean(np.abs(predictions - labels))


# given a list of predictions and a list of labels, return the RMSE
@numpify
def RMSE(predictions, labels):
    return np.sqrt(np.mean((predictions - labels) ** 2))


# given a list of predictions and a list of labels, return the MAPE
@numpify
def MAPE(predictions, labels):
    return np.mean(np.abs((predictions - labels) / labels)) * 100


# given a list of predictions and a list of labels, return the R2 score
@numpify
def R2(predictions, labels):
    return 1 - np.sum((predictions - labels) ** 2) / np.sum(
        (labels - np.mean(labels)) ** 2
    )


# given a list of predictions and a list of labels, return a table of metrics
def metrics_table(predictions, labels, return_scores=False):
    # mse = MSE(predictions, labels)
    mae = MAE(predictions, labels)
    rmse = RMSE(predictions, labels)
    mape = MAPE(predictions, labels)
    # r2 = R2(predictions, labels)

    # Center the strings
    # mse_str = f"{mse:.3f}".center(7)
    mae_str = f"{mae:.3f}".center(7)
    rmse_str = f"{rmse:.3f}".center(7)
    mape_str = f"{mape:.3f}".center(7)
    # r2_str = f"{r2:.3f}".center(7)

    if not return_scores:
        return (
            f"+-------+-------+-------+\n"
            f"|  MAE  |  RMSE |  MAPE |\n"
            f"+-------+-------+-------+\n"
            f"|{mae_str}|{rmse_str}|{mape_str}|\n"
            f"+-------+-------+-------+"
        )
    else:
        return (
            f"+-------+-------+-------+\n"
            f"|  MAE  |  RMSE |  MAPE |\n"
            f"+-------+-------+-------+\n"
            f"|{mae_str}|{rmse_str}|{mape_str}|\n"
            f"+-------+-------+-------+"
        ), {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        }
