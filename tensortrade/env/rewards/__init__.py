from tensortrade.env.rewards.abstract import TensorTradeRewardScheme
from tensortrade.env.rewards.simple_profit import SimpleProfit
from tensortrade.env.rewards.risk_adjusted_returns import RiskAdjustedReturns
from tensortrade.env.rewards.pbr import PBR

_registry = {
    'simple': SimpleProfit,
    'risk-adjusted': RiskAdjustedReturns,
    'pbr': PBR,
}


def get(identifier: str) -> TensorTradeRewardScheme:
    """Gets the `RewardScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `RewardScheme`

    Returns
    -------
    `TensorTradeRewardScheme`
        The reward scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry.keys():
        msg = f"Identifier {identifier} is not associated with any `RewardScheme`."
        raise KeyError(msg)
    return _registry[identifier]()