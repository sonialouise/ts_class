import argparse
import logging
import pickle
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble, TimeSeriesForestClassifier
from math import floor
from .config import OBS, LABEL, TSF_MODEL, RISE_MODEL
from .prepare_data import prepare_data

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def split_train_test(data):
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(data[OBS]), data[LABEL], test_size=0.2)
    print(f"Split data shapes: X_train: {X_train.shape}\nY_train: {y_train.shape}\nX_test: {X_test.shape}\nY_test: {y_test.shape}")

    labels_counts = zip(np.unique(y_train, return_counts=True))
    print(f"Data contains: {labels_counts}")
    return X_train, X_test, y_train, y_test


def create_model_performance_summary(model_summary, y_test, y_pred_test):
    model_summary['Accuracy'].append(accuracy_score(y_test, y_pred_test))
    model_summary['Precision'].append(
        precision_score(y_test, y_pred_test, average='binary', pos_label='active'))
    model_summary['Recall'].append(recall_score(y_test, y_pred_test, average='binary', pos_label='active'))
    return model_summary


def train_and_cross_validate(data, model, model_name):
    kf5 = KFold(n_splits=5, shuffle=True)
    model_summary = defaultdict(list)
    for train_index, test_index in kf5.split(data):
        X_train = pd.DataFrame(data.iloc[train_index].loc[:, OBS])
        X_test = pd.DataFrame(data.iloc[test_index][OBS])
        y_train = data.iloc[train_index].loc[:, LABEL]
        y_test = data.iloc[test_index][LABEL]

        # Train model
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        model_summary = create_model_performance_summary(model_summary, y_test, y_pred_test)
    average_performance = {k.upper(): np.mean(v) for k, v in model_summary.items()}

    print(f"RESULTS {model_name}: {average_performance}")
    return model


def prepare_timeseries_forest_classifier():
    tsf = TimeSeriesForestClassifier(
        n_estimators=10
    )
    return tsf


def prepare_random_interval_spectral_ensemble(n_estimators=10):
    return RandomIntervalSpectralEnsemble(n_estimators=n_estimators)


def retrain_tsf(data):
    tsf = prepare_timeseries_forest_classifier()
    tsf = train_and_cross_validate(data, tsf, 'TSF')
    pickle.dump(tsf, open(TSF_MODEL, 'wb'))
    return tsf


def retrain_rise(data):
    rise = prepare_random_interval_spectral_ensemble()
    rise = train_and_cross_validate(data, rise, 'RISE')
    pickle.dump(rise, open(RISE_MODEL, 'wb'))


def retrain_models(args):
    logging.info("1. Reading csv file...")
    data = pd.read_csv(args.data, header=0)
    logging.info("2. Normalising data...")
    train_total_frames = floor(args.duration * args.sampling_rate)
    trace_data = deepcopy(data.iloc[:, 0:train_total_frames])
    data = pd.concat([prepare_data(trace_data, args.duration, args.sampling_rate), data[LABEL]], axis=1)
    logging.info("3. Retraining Time Series Classifier...")
    retrain_tsf(data)
    logging.info("4. Retraining Random Interval Spectral Ensemble...")
    retrain_rise(data)
    logging.info("5. Retraining Complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('data', help=f'Data to retrain model (csv format, must contain {LABEL} column)')
    parser.add_argument('duration', type=int, help=f'Total recording duration in seconds')
    parser.add_argument('sampling_rate', type=float, help=f'Recording sampling rate (Hz)')
    args = parser.parse_args()

    retrain_models(args)
