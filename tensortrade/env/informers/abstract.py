import typing
from abc import abstractmethod

from tensortrade.env.generic import Informer

if typing.TYPE_CHECKING:
    from typing import Any, Dict

    from tensortrade.env.generic import TradingEnv

class TensorTradeInformer(Informer):
    @abstractmethod
    def info(self, env: TradingEnv) -> Dict[str, Any]:
        raise NotImplementedError()
