# Copyright 2024 The TensorTrade and TensorTrade-NG Authors.
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
from abc import abstractmethod

from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed
from tensortrade.env.mixins.scheme import SchemeMixin

if typing.TYPE_CHECKING:
    from typing import List

    from gymnasium.core import ActType
    from gymnasium.spaces import Space

    from tensortrade.oms.orders import Order

class AbstractActionScheme(SchemeMixin, Component, TimeIndexed):
    """An abstract base class for any `ActionScheme` that wants to be
    compatible with the built-in OMS.

    The structure of the action scheme is built to make sure that action space
    can be used with the system, provided that the user defines the methods to
    interpret that action.
    """

    registered_name = "action_scheme"

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the :class:`TradingEnv`.

        :return: The gymnasium action space of the :class:`TradingEnv`.
        :rtype: Space
        """
        raise NotImplementedError()

    @abstractmethod
    def get_orders(self, action: ActType) -> List[Order]:
        """Gets the list of orders to be submitted for the given action.

        :param action: The action to be interpreted.
        :type action: :class:`gymnasium.core.ActType`
        :return: A list of orders to be submitted to the broker.
        :rtype: List[Order]
        """
        raise NotImplementedError()

    def perform_action(self, action: ActType) -> None:
        """Performs the action on the given environment.

        :param action: The action selected by the agent to perform on the environment.
        :type action: ActType
        """
        orders = self.get_orders(action)

        for order in orders:
            if order:
                self.trading_env.broker.submit(order)

        self.trading_env.broker.update()

    def reset(self) -> None:
        """Performs a reset on the action scheme. Will be called when :class:`TradingEnv` is reset.
        """
        pass