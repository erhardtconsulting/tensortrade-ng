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
from itertools import product

from gymnasium.spaces import Discrete

from tensortrade.env.interfaces import AbstractActionScheme
from tensortrade.oms.orders import TradeType, TradeSide

if typing.TYPE_CHECKING:
    from typing import List, Optional, Union

    from gymnasium import Space

    from tensortrade.oms.orders import Order, OrderListener
    from tensortrade.oms.orders.criteria import Criteria
    from tensortrade.oms.wallets import Portfolio


class SimpleOrders(AbstractActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes.

    Parameters
    ----------
    criteria : List[OrderCriteria]
        A list of order criteria to select from when submitting an order.
        (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
    trade_sizes : List[float]
        A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
        '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
    durations : List[int]
        A list of durations to select from when submitting an order.
    trade_type : TradeType
        A type of trade to make.
    order_listener : OrderListener
        A callback class to use for listening to steps of the order process.
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    registered_name = "simple_orders"

    def __init__(self,
                 criteria: Union[List[Criteria], Criteria] = None,
                 trade_sizes: Union[List[float], int] = 10,
                 durations: Union[List[int], int] = None,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: Optional[OrderListener] = None,
                 min_order_pct: float = 0.02,
                 min_order_abs: float = 0.00) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        criteria = self.default('criteria', criteria)
        self.criteria = criteria if isinstance(criteria, list) else [criteria]

        trade_sizes = self.default('trade_sizes', trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default('durations', durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> Space:
        if not self._action_space:
            self.actions = product(
                self.criteria,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL]
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self,
                   action: int,
                   portfolio: Portfolio) -> List[Order]:

        if action == 0:
            return []

        (ep, (criteria, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)

        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        order = Order(
            step=self.clock.step,
            side=side,
            trade_type=self._trade_type,
            exchange_pair=ep,
            price=ep.price,
            quantity=quantity,
            criteria=criteria,
            end=self.clock.step + duration if duration else None,
            portfolio=portfolio
        )

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]
