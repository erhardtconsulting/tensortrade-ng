# Copyright 2024 The TensorTrade-NG Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from __future__ import annotations

import typing

from tensortrade.env.actions.bsh import BSH
from tensortrade.env.actions.simple_orders import SimpleOrders
from tensortrade.env.actions.managed_risk_orders import ManagedRiskOrders

if typing.TYPE_CHECKING:
    from tensortrade.env import AbstractActionScheme

_registry = {
    'bsh': BSH,
    'simple': SimpleOrders,
    'managed-risk': ManagedRiskOrders,
}


def get(identifier: str) -> AbstractActionScheme:
    """
    Gets the :class:`AbstractActionScheme` that matches with the identifier.

    :param identifier: The identifier for the ``AbstractActionScheme``
    :type identifier: str
    :returns: The action scheme associated with the ``identifier``.
    :rtype: class:`AbstractActionScheme`
    :raises KeyError: Raised if identifier is not associated with any :class:`AbstractActionScheme`
    """
    if identifier not in _registry.keys():
        msg = f"Identifier {identifier} is not associated with any `RewardScheme`."
        raise KeyError(msg)
    return _registry[identifier]()