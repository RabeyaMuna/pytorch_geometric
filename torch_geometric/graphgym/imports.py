import warnings

import torch

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            from pyparsing import PyparsingDeprecationWarning
            warnings.simplefilter('ignore', PyparsingDeprecationWarning)
        except ImportError:
            pass
        try:
            import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import pytorch_lightning as pl
        except Exception:
            raise ImportError
    _pl_is_available = True
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import lightning.pytorch as pl
        _pl_is_available = True
    except ImportError:
        _pl_is_available = False

if _pl_is_available:
    LightningModule = pl.LightningModule
    Callback = pl.Callback
else:
    pl = object
    LightningModule = torch.nn.Module
    Callback = object

    warnings.warn(
        "To use GraphGym, install 'pytorch_lightning' or 'lightning' via "
        "'pip install pytorch_lightning' or 'pip install lightning'",
        stacklevel=2)
