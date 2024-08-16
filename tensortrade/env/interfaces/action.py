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

import logging
import typing
from abc import abstractmethod

from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed
from tensortrade.oms.orders import Broker

if typing.TYPE_CHECKING:
    from typing import List, Any, Optional

    from gymnasium.spaces import Space

    from tensortrade.core import Clock
    from tensortrade.env.interfaces import TradingEnv
    from tensortrade.oms.orders import Order
    from tensortrade.oms.wallets import Portfolio

class AbstractActionScheme(Component, TimeIndexed):
    """An abstract base class for any `ActionScheme` that wants to be
    compatible with the built in OMS.

    The structure of the action scheme is built to make sure that action space
    can be used with the system, provided that the user defines the methods to
    interpret that action.

    Attributes
    ----------
    portfolio : Portfolio
        The portfolio object to be used in defining actions.
    broker : Broker
        The broker object to be used for placing orders in the OMS.

    Methods
    -------
    perform(env,portfolio)
        Performs the action on the given environment.
    get_orders(action,portfolio)
        Gets the list of orders to be submitted for the given action.
    """

    registered_name = "abstract_action_scheme"

    def __init__(self) -> None:
        super().__init__()
        self.portfolio: Optional[Portfolio] = None
        self.broker: Broker = Broker()

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()

    @property
    def clock(self) -> Clock:
        """The reference clock from the environment. (`Clock`)

        When the clock is set for the we also set the clock for the portfolio
        as well as the exchanges defined in the portfolio.

        Returns
        -------
        `Clock`
            The environment clock.
        """
        return self._clock

    @clock.setter
    def clock(self, clock: Clock) -> None:
        self._clock = clock

        components = [self.portfolio] + self.portfolio.exchanges
        for c in components:
            c.clock = clock
        self.broker.clock = clock

    def perform(self, env: TradingEnv, action: Any) -> None:
        """Performs the action on the given environment.

        Under the TT action scheme, the subclassed action scheme is expected
        to provide a method for getting a list of orders to be submitted to
        the broker for execution in the OMS.

        Parameters
        ----------
        env : 'TradingEnv'
            The environment to perform the action on.
        action : Any
            The specific action selected from the action space.
        """
        orders = self.get_orders(action, self.portfolio)

        for order in orders:
            if order:
                logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                self.broker.submit(order)

        self.broker.update()

    @abstractmethod
    def get_orders(self, action: Any, portfolio: Portfolio) -> List[Order]:
        """Gets the list of orders to be submitted for the given action.

        Parameters
        ----------
        action : Any
            The action to be interpreted.
        portfolio : Portfolio
            The portfolio defined for the environment.

        Returns
        -------
        List[Order]
            A list of orders to be submitted to the broker.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        self.portfolio.reset()
        self.broker.reset()