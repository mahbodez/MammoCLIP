from typing import List
from torch import nn
from logging import getLogger

logger = getLogger(__name__)


def freeze_params(model: nn.Module, freeze: bool = True) -> None:
    """
    Freezes or unfreezes the parameters of a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model whose parameters will be modified.
        freeze (bool, optional): If True, all parameters will be frozen (requires_grad=False).
            If False, all parameters will be unfrozen (requires_grad=True). Defaults to True.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad_(not freeze)

    logger.info(
        f"Parameters {'frozen' if freeze else 'unfrozen'} for model: {model.__class__.__name__}"
    )


def freeze_submodules(
    parent_model: nn.Module, submodules: List[str], freeze: bool = True
) -> None:
    """
    Freezes or unfreezes the parameters of specific submodules within a parent model.

    Args:
        parent_model (nn.Module): The parent model containing the submodules.
        submodules (List[str]): List of submodules to be frozen/unfrozen.
        freeze (bool, optional): If True, the specified submodules will be frozen.
            If False, they will be unfrozen. Defaults to True.

    Returns:
        None
    """
    for submodule in submodules:
        if hasattr(parent_model, submodule):
            freeze_params(getattr(parent_model, submodule), freeze)
        else:
            logger.error(
                f"Submodule {submodule} is not a child of the parent model {parent_model}."
            )
            raise ValueError(
                f"Submodule {submodule} is not a child of the parent model {parent_model}."
            )
    logger.info(
        f"Submodules {'frozen' if freeze else 'unfrozen'} for model: {parent_model.__class__.__name__}"
    )
