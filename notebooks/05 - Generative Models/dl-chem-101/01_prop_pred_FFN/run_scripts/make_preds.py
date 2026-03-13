""" Make predictions
"""

import pred_ffn.predict as predict
import logging
import time

if __name__ == "__main__":
    start_time = time.time()
    predict.predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
