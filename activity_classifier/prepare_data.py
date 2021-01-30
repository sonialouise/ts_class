from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from activity_classifier.config import OBS
import numpy as np


def normalise_data(data):
    assert data.isnull().sum().sum() == 0, AssertionError("Data contains empty values, correct and retry")
    assert np.isinf(data).values.sum() == 0, AssertionError("Data contains inf values, correct and retry")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    data[OBS] = [pd.Series(row) for row in scaled]
    return data