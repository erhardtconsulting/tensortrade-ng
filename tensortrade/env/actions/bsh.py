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

from gymnasium.spaces import Discrete

from tensortrade.env.interfaces import AbstractActionScheme
from tensortrade.oms.orders import proportion_order

if typing.TYPE_CHECKING:
    from typing import List

    from tensortrade.oms.orders import Order
    from tensortrade.oms.wallets import Portfolio, Wallet


class BSH(AbstractActionScheme):
    """A simple discrete action scheme where the only options are to buy, sell,
    or hold.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "bsh"

    def __init__(self, cash: Wallet, asset: Wallet):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: Portfolio) -> List[Order]:
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash

            if src.balance == 0:  # We need to check, regardless of the proposed order, if we have balance in 'src'
                return []  # Otherwise just return an empty order list

            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0