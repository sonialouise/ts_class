import pandas as pd
from activity_classifier.config import OBS
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""Scaling/normalisation to use as required"""
scaler = StandardScaler() # MinMaxScaler()

def normalise_data(data):
    assert data.isnull().sum().sum() == 0, AssertionError("Data contains empty values, correct and retry")
    assert np.isinf(data).values.sum() == 0, AssertionError("Data contains inf values, correct and retry")
    scaled = scaler.fit_transform(np.array(data))
    data[OBS] = [pd.Series(row) for row in scaled]
    return data


def prepare_data(data):
    assert data.isnull().sum().sum() == 0, AssertionError("Data contains empty values, correct and retry")
    assert np.isinf(data).values.sum() == 0, AssertionError("Data contains inf values, correct and retry")
    data[OBS] = [pd.Series(row) for row in np.array(data)]
    return data