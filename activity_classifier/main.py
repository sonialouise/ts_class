import pickle
from .config import TSF_MODEL, RISE_MODEL, OBS, PREDICTION, OUTPUT_PATH
from .prepare_data import normalise_data
import argparse
import logging
import pandas as pd

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def run_model(data, model, model_name):
    loaded_model = pickle.load(open(model, 'rb'))
    result = loaded_model.predict(pd.DataFrame(data[OBS]))
    data[f"{model_name}_{PREDICTION}"] = result
    return data


def main(data):
    logging.info("1. Reading csv file...")
    data = pd.read_csv(data)
    logging.info("2. Normalising data...")
    data = normalise_data(data)
    logging.info("3. Predicting with TimeSeries Forest Classifier...")
    data = run_model(data, TSF_MODEL, 'TSF')
    logging.info("4. Predicting with Random Interval Spectral Ensemble...")
    data = run_model(data, RISE_MODEL, 'RISE')
    logging.info("5. Saving output...")
    data.to_csv(OUTPUT_PATH)
    logging.info("6. Process complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('data', help=f'Data to retrain model (csv format)')
    args = parser.parse_args()

    main(args.data)
