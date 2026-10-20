import importlib
from typing import TYPE_CHECKING

__all__ = classes = [
    'LightningDataset',
    'LightningNodeData',
    'LightningLinkData',
]

if TYPE_CHECKING:
    from .datamodule import (LightningDataset, LightningLinkData,
                             LightningNodeData)


def __getattr__(name: str):
    if name in classes:
        from . import datamodule
        return getattr(datamodule, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


