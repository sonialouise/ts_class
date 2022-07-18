from pathlib import Path
from config import TSF_MODEL, RISE_MODEL, OBS, PREDICTION, OUTPUT_PATH, LABEL
from prepare_data import prepare_data
from main import run_model
from retrain_models import (
    retrain_rise,
    prepare_timeseries_forest_classifier,
    prepare_random_interval_spectral_ensemble,
    train_and_cross_validate
)
import logging
import pandas as pd
import pickle

path = Path('.')

train_data = path.cwd().parent / 'data' / 'training_data.csv'
test_data = path.cwd().parent / 'data' / 'test_set.csv'

## PREDICT
data = pd.read_csv(test_data, header=0)
logging.info("2. Normalising data...")
data = prepare_data(data.iloc[:, 0:550])
logging.info("3. Predicting with TimeSeries Forest Classifier...")
data = run_model(data, path.cwd().parent / TSF_MODEL, 'TSF')
logging.info("4. Predicting with Random Interval Spectral Ensemble...")
data = run_model(data, path.cwd().parent / RISE_MODEL, 'RISE')
logging.info("5. Saving output...")
data.to_csv(OUTPUT_PATH)


## RETRAIN
# logging.info("1. Reading csv file...")
# data = pd.read_csv(train_data, header=0)
# logging.info("2. Normalising data...")
# data = pd.concat([prepare_data(data.iloc[:, 0:int(550)]), data[LABEL]], axis=1)
# logging.info("3. Retraining Time Series Classifier...")
# tsf = prepare_timeseries_forest_classifier()
# tsf = train_and_cross_validate(data, tsf, 'TSF')
# pickle.dump(tsf, open(path.cwd().parent / TSF_MODEL, 'wb'))
# logging.info("4. Retraining Random Interval Spectral Ensemble...")
# rise = prepare_random_interval_spectral_ensemble()
# rise = train_and_cross_validate(data, rise, 'RISE')
# pickle.dump(rise, open(path.cwd().parent / RISE_MODEL, 'wb'))
# logging.info("5. Retraining Complete")