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

from tensortrade.env.interfaces import AbstractRewardScheme
from tensortrade.feed import DataFeed, Stream

if typing.TYPE_CHECKING:
    from tensortrade.oms.wallets import Portfolio

class PBR(AbstractRewardScheme):
    """A reward scheme for position-based returns.

    * Let :math:`p_t` denote the price at time t.
    * Let :math:`x_t` denote the position at time t.
    * Let :math:`R_t` denote the reward at time t.

    Then the reward is defined as,
    :math:`R_{t} = (p_{t} - p_{t-1}) \cdot x_{t}`.

    Parameters
    ----------
    price : `Stream`
        The price stream to use for computing rewards.
    """

    registered_name = "pbr"

    def __init__(self, price: Stream) -> None:
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (position * r).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        self.position = -1 if action == 0 else 1

    def reward(self, portfolio: Portfolio) -> float:
        return self.feed.next()["reward"]

    def reset(self) -> None:
        """Resets the `position` and `feed` of the reward scheme."""
        self.position = -1
        self.feed.reset()