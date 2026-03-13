""" Train a model.
"""

import pred_gnn.train as train
import logging
import time

if __name__=="__main__": 
    start_time = time.time()
    train.train_model()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
