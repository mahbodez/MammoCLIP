from custom.config import Config
import os
import re
from typing import Tuple


def find_latest_checkpoint(project_dir: str) -> Tuple[int, str]:
    """
    Returns (latest_epoch, checkpoint_dir) or (0, "") if none found.
    """
    model_dirs = [
        f
        for f in os.listdir(project_dir)
        if re.match(r"(model_resumed|model)_(\d+)$", f)
        and os.path.isdir(os.path.join(project_dir, f))
    ]
    if not model_dirs:
        print("No previous epochs found.")
        return 0, ""
    most_recent = max(model_dirs, key=lambda d: int(re.search(r"\d+", d).group()))
    epoch = int(re.search(r"\d+", most_recent).group())
    return epoch, os.path.join(project_dir, most_recent)
