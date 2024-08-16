from tensortrade.env.actions.abstract import TensorTradeActionScheme
from tensortrade.env.actions.bsh import BSH
from tensortrade.env.actions.simple_orders import SimpleOrders
from tensortrade.env.actions.managed_risk_orders import ManagedRiskOrders
from tensortrade.env.generic import ActionScheme

_registry = {
    'bsh': BSH,
    'simple': SimpleOrders,
    'managed-risk': ManagedRiskOrders,
}


def get(identifier: str) -> ActionScheme:
    """Gets the `ActionScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `ActionScheme`.

    Returns
    -------
    ActionScheme
        The action scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if the `identifier` is not associated with any `ActionScheme`.
    """
    if identifier not in _registry.keys():
        raise KeyError(f"Identifier {identifier} is not associated with any `ActionScheme`.")
    return _registry[identifier]()