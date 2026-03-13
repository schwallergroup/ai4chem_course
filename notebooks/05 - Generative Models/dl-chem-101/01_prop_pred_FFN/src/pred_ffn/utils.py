""" utils.py """
import sys
import copy
import logging
from pathlib import Path

from tqdm import tqdm

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.base import rank_zero_experiment


def setup_logger(save_dir, log_name="output.log", debug=False):
    """setup_logger.

    Args:
        save_dir:
        log_name:
        debug:
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    log_file = save_dir / log_name

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


class ConsoleLogger(LightningLoggerBase):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        ## No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


# Parallel operations
def simple_parallel(input_list,
                    function,
                    max_cpu=16,
                    timeout=4000,
                    max_retries=3):
    """simple_parallel.

    Use map async and retries in case we get odd stalling behavior.

    Args:
        input_list:
        function:
        max_cpu:
        timeout:
        max_retries:
    """
    from multiprocess.context import TimeoutError
    from pathos import multiprocessing as mp

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    async_results = [
        pool.apply_async(function, args=(i, )) for i in input_list
    ]
    pool.close()

    retries = 0
    while True:
        try:
            list_outputs = []
            for async_result in tqdm(async_results, total=len(input_list)):
                result = async_result.get(timeout)
                list_outputs.append(result)

            break
        except TimeoutError:
            retries += 1
            logging.info(f"Timeout Error (s > {timeout})")
            if retries <= max_retries:
                pool = mp.Pool(processes=cpus)
                async_results = [
                    pool.apply_async(function, args=(i, )) for i in input_list
                ]
                pool.close()
                logging.info(f"Retry attempt: {retries}")
            else:
                raise ValueError()

    return list_outputs


def chunked_parallel(input_list,
                     function,
                     chunks=100,
                     max_cpu=16,
                     timeout=4000,
                     max_retries=3):
    """chunked_parallel.

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
        timeout: Length of timeout
        max_retries: Num times to retry this
    """

    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i:i + step_size]
        for i in range(0, len(input_list), step_size)
    ]

    list_outputs = simple_parallel(chunked_list,
                                   batch_func,
                                   max_cpu=max_cpu,
                                   timeout=timeout,
                                   max_retries=max_retries)
    # Unroll
    full_output = [j for i in list_outputs for j in i]

    return full_output
