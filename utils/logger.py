import logging
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
