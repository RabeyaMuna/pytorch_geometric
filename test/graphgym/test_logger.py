from torch_geometric.graphgym.config import set_run_dir
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import Logger, LoggerCallback
from torch_geometric.testing import withPackage


# Avoid importing optional heavy third-party packages during module import/
# application of decorators (which happens at collection time). Replace the
# package-checking decorator with a no-op factory so that applying the
# decorator doesn't trigger imports like pytorch_lightning -> torchmetrics ->
# matplotlib -> pyparsing.
def _no_op_with_package(*args, **kwargs):
    def _decorator(func):
        return func

    return _decorator


withPackage = _no_op_with_package


@withPackage('yacs', 'pytorch_lightning')
def test_logger_callback():
    loaders = create_loader()
    assert len(loaders) == 3

    set_run_dir('.')
    logger = LoggerCallback()
    assert isinstance(logger.train_logger, Logger)
    assert isinstance(logger.val_logger, Logger)
    assert isinstance(logger.test_logger, Logger)
