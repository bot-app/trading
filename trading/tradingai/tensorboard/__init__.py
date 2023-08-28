# ensure users can still use a non-torch tradingai version
try:
    from trading.tradingai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger
    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from trading.tradingai.tensorboard.base_tensorboard import (BaseTensorBoardCallback,
                                                               BaseTensorboardLogger)
    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = (
    "TBLogger",
    "TBCallback"
)
