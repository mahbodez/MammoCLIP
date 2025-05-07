import logging
from logging import Logger
from custom.config import Config
from torch.utils.tensorboard import SummaryWriter
from utils.dist_utils import is_main_process
from typing import Optional
import os


def setup_logger(
    name: str, level: int = logging.INFO, log_to_file: Optional[str] = None
):
    """
    Sets up a logger with the given name and level.

    Parameters:
        name (str): Logger name.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_file (str, optional): Path to the log file. If None, logs only to console.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional: File Handler
    if log_to_file:
        os.makedirs(os.path.dirname(log_to_file), exist_ok=True)
        fh = logging.FileHandler(log_to_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def init_logger(config: Config) -> Logger:
    """
    Initialize logger on main process.
    """
    # Setup logger only on main process
    logger = None
    if is_main_process():
        os.makedirs(config.project_dir, exist_ok=True)
        logger = setup_logger(
            "training",
            log_to_file=os.path.join(config.project_dir, "logs", "train.log"),
        )
    return logger


def init_tensorboard(config: Config) -> SummaryWriter | None:
    """
    Initialize TensorBoard writer on main process.
    """
    tb_writer = None
    if is_main_process():
        path = os.path.join(config.project_dir, "logs", "tensorboard")
        os.makedirs(path, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=path)
    return tb_writer
