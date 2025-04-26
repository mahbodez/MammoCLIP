import math
import re
import subprocess
import torch
from typing import Dict


def stats_from_epochs(
    num_epochs: int,
    num_gpus: int,
    accumulation_steps: int,
    per_gpu_batch_size: int,
    dataset_size: int,
) -> Dict[str, float]:
    """
    Compute training statistics for DDP + gradient accumulation.

    Args:
      num_epochs: total epochs to run
      num_gpus: number of processes (GPUs)
      accumulation_steps: how many forward/backward calls before each optimizer.step()
      per_gpu_batch_size: batch size seen by each process
      dataset_size: total samples in the Dataset

    Returns:
      A dict containing:
        num_epochs
        effective_batch_size: per‐step global batch size = per_gpu_bs * num_gpus * accumulation_steps
        iterations_per_epoch: number of DataLoader batches per process = ceil(dataset_size / per_gpu_bs)
        total_batches: total forward/backward calls across all epochs = iterations_per_epoch * num_epochs
        optimization_steps_per_epoch: number of optimizer.step calls per epoch = ceil(iterations_per_epoch / accumulation_steps)
        total_optimization_steps: total optimizer.step calls = optimization_steps_per_epoch * num_epochs
        samples_per_optimization_step: number of samples processed between updates = effective_batch_size
        samples_per_epoch: total samples processed per epoch across all GPUs = iterations_per_epoch * per_gpu_bs * num_gpus
        total_samples: total samples processed across all epochs = samples_per_epoch * num_epochs
    """
    # how many samples processed in one optimizer.step()
    effective_batch_size = per_gpu_batch_size * num_gpus * accumulation_steps

    # batches per epoch seen by each process (GPU)
    iterations_per_epoch = math.ceil(dataset_size / per_gpu_batch_size)

    # total forward/backward passes (batches) across all epochs, per process
    total_batches = iterations_per_epoch * num_epochs

    # how many optimizer updates per epoch
    optimization_steps_per_epoch = math.ceil(iterations_per_epoch / accumulation_steps)

    # total optimizer updates over all epochs
    total_optimization_steps = optimization_steps_per_epoch * num_epochs

    # total samples processed per epoch across all GPUs
    samples_per_epoch = iterations_per_epoch * per_gpu_batch_size * num_gpus

    # total samples processed across the entire run
    total_samples = samples_per_epoch * num_epochs

    return {
        "num_epochs": num_epochs,
        "effective_batch_size": effective_batch_size,
        "iterations_per_epoch": iterations_per_epoch,
        "total_batches": total_batches,
        "optimization_steps_per_epoch": optimization_steps_per_epoch,
        "total_optimization_steps": total_optimization_steps,
        "samples_per_optimization_step": effective_batch_size,
        "samples_per_epoch": samples_per_epoch,
        "total_samples": total_samples,
    }


def stats_from_steps(
    total_optimization_steps,
    num_gpus,
    accumulation_steps,
    per_gpu_batch_size,
    dataset_size,
):
    """
    Computes training statistics for distributed training with gradient accumulation.

    Parameters:
    - total_optimization_steps (int): Desired total number of optimizer (parameter update) steps.
    - num_gpus (int): Number of GPUs used in training.
    - accumulation_steps (int): Number of gradient accumulation steps.
    - per_gpu_batch_size (int): Batch size per GPU.
    - dataset_size (int): Total number of samples in the dataset.

    Returns:
    - dict: A dictionary containing computed training statistics.
    """
    # Effective batch size across all GPUs and accumulation steps
    effective_batch_size = per_gpu_batch_size * num_gpus * accumulation_steps

    # Samples processed per optimization step
    samples_per_optimization_step = effective_batch_size

    # Total iterations (total number of forward/backward passes)
    total_iterations = total_optimization_steps * accumulation_steps

    # Iterations per epoch (number of batches per epoch)
    iterations_per_epoch = math.ceil(dataset_size / (per_gpu_batch_size * num_gpus))

    # Number of epochs required to complete total_iterations
    num_epochs = math.ceil(total_iterations / iterations_per_epoch)

    # Adjust iterations_per_epoch if num_epochs * iterations_per_epoch exceeds total_iterations
    adjusted_iterations_per_epoch = iterations_per_epoch
    if num_epochs * iterations_per_epoch > total_iterations:
        adjusted_iterations_per_epoch = math.ceil(total_iterations / num_epochs)

    # Optimization steps per epoch
    optimization_steps_per_epoch = math.ceil(
        adjusted_iterations_per_epoch / accumulation_steps
    )

    # Samples per epoch
    samples_per_epoch = adjusted_iterations_per_epoch * per_gpu_batch_size * num_gpus

    # Total samples processed
    total_samples = (
        total_iterations * per_gpu_batch_size * num_gpus / accumulation_steps
    )

    # Prepare the results dictionary
    results = {
        "num_epochs": num_epochs,
        "effective_batch_size": effective_batch_size,
        "iterations_per_epoch": adjusted_iterations_per_epoch,
        "total_iterations": total_iterations,
        "optimization_steps_per_epoch": optimization_steps_per_epoch,
        "total_optimization_steps": total_optimization_steps,
        "samples_per_optimization_step": samples_per_optimization_step,
        "samples_per_epoch": samples_per_epoch,
        "total_samples": total_samples,
    }

    return results


def get_gpu_memory_usage():
    """
    Get the combined GPU memory usage using nvidia-smi for accuracy.

    Returns:
    dict: A dictionary containing total used, total memory, and utilization percentage across all GPUs.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA is not available."}

    # Get real GPU memory usage using nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Parse nvidia-smi output for all GPUs
        memory_data = result.stdout.strip().split("\n")
        total_used = 0
        total_memory = 0

        for line in memory_data:
            used_memory, total_gpu_memory = map(int, re.findall(r"\d+", line))
            total_used += used_memory
            total_memory += total_gpu_memory

        utilization_percentage = (
            (total_used / total_memory) * 100 if total_memory > 0 else 0
        )

        return {
            "used": total_used,
            "total": total_memory,
            "percentage": utilization_percentage,
        }

    except Exception:
        return None


def get_gpu_power_usage():
    """
    Get the combined GPU power usage using nvidia-smi for accuracy.

    Returns:
    dict: A dictionary containing total power usage in watts across all GPUs.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA is not available."}

    # Get real GPU power usage using nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,nounits,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Parse nvidia-smi output for all GPUs
        power_data = result.stdout.strip().split("\n")
        total_power = 0

        for line in power_data:
            power = float(line)
            total_power += power

        return {"power": total_power}

    except Exception:
        return None


def get_gpu_temperature():
    """
    Get the combined GPU temperature using nvidia-smi for accuracy.

    Returns:
    dict: A dictionary containing total temperature in degrees Celsius across all GPUs.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA is not available."}

    # Get real GPU temperature using nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,nounits,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Parse nvidia-smi output for all GPUs
        temperature_data = result.stdout.strip().split("\n")
        total_temperature = 0

        for line in temperature_data:
            temperature = int(line)
            total_temperature += temperature

        return {"temp": total_temperature / len(temperature_data)}

    except Exception:
        return None
