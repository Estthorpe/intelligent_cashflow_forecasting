import pandas as pd

def time_split(daily: pd.DataFrame, train_end: str, val_end: str):
    train = daily[daily["ds"] <= train_end]
    val   = daily[(daily["ds"] > train_end) & (daily["ds"] <= val_end)]
    test  = daily[daily["ds"] > val_end]
    return train, val, test

def mape(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    denom = y_true.replace(0, 1e-9).abs()
    return (abs((y_true - y_pred) / denom)).mean() * 100

def rmse(y_true, y_pred):
    diff = (pd.Series(y_true) - pd.Series(y_pred)) ** 2
    return (diff.mean()) ** 0.5
