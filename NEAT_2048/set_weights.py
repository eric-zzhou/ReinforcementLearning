import pandas
from pandas import DataFrame
import logging

import pickle

with open(r'/NEAT_2048/weights.pkl', 'rb') as f:
    weights = pickle.load(f)

logging.info(weights)

pickle.dump([0.875, 0.625, 1, 0.625, 0.05, 2], open(r'/NEAT_2048/weights.pkl', 'wb'))

# WEIGHTS_COR = ["emptyw", "edgew", "smoothw", "matchw", "monow", "snakew"]
# logging.info(WEIGHTS_COR[1:2])
