from pathlib import Path
from activity_classifier.config import TSF_MODEL, RISE_MODEL, OUTPUT_PATH
from activity_classifier.prepare_data import prepare_data
from activity_classifier.main import run_model
from retrain_models import retrain_rise, retrain_tsf
import logging
import pandas as pd

path = Path('.')

train_data = path.cwd().parent / 'data' / 'training_data.csv'
test_data = path.cwd().parent / 'data' / 'test_data.csv'

duration = 151
sampling_rate = 3.65

## PREDICT
logging.info("1. Reading csv file...")
data = pd.read_csv(test_data, header=0)
logging.info("2. Normalising data...")
total_frames = int(duration * sampling_rate)
trace_data = data.iloc[:, 0:total_frames]
data = prepare_data(trace_data, duration, sampling_rate)
logging.info("3. Predicting with TimeSeries Forest Classifier...")
data = run_model(data, TSF_MODEL, 'TSF')
logging.info("4. Predicting with Random Interval Spectral Ensemble...")
data = run_model(data, RISE_MODEL, 'RISE')
logging.info("5. Saving output...")
data.to_csv(OUTPUT_PATH)
logging.info("6. Process complete")


## RETRAIN
logging.info("1. Reading csv file...")
data = pd.read_csv(train_data, header=0)
logging.info("2. Normalising data...")
total_frames = int(duration * sampling_rate)
trace_data = data.iloc[:, 0:total_frames]
data = pd.concat([prepare_data(trace_data, duration, sampling_rate), data['status']], axis=1)
logging.info("3. Retraining Time Series Classifier...")
retrain_tsf(data)
logging.info("4. Retraining Random Interval Spectral Ensemble...")
retrain_rise(data)
logging.info("5. Retraining Complete")