import typing

from tensortrade.env.generic import Informer

if typing.TYPE_CHECKING:
    from typing import Any, Dict
    from tensortrade.env.generic import TradingEnv

class SimpleInformer(Informer):
    def info(self, env: TradingEnv) -> Dict[str, Any]:
        return {
            'step': self.clock.step,
            'net_worth': env.action_scheme.portfolio.net_worth
        }