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

if typing.TYPE_CHECKING:
    from tensortrade.oms.wallets import Portfolio

class SimpleProfit(AbstractRewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases
    in net worth.

    Parameters
    ----------
    window_size : int
        The size of the look back window for computing the reward.

    Attributes
    ----------
    window_size : int
        The size of the look back window for computing the reward.
    """

    registered_name = "simple_profit"

    def __init__(self, window_size: int = 1):
        self._window_size = self.default('window_size', window_size)

    def reward(self, portfolio: Portfolio) -> float:
        """Rewards the agent for incremental increases in net worth over a
        sliding window.

        Parameters
        ----------
        portfolio : `Portfolio`
            The portfolio being used by the environment.

        Returns
        -------
        float
            The cumulative percentage change in net worth over the previous
            `window_size` time steps.
        """
        net_worths = [nw['net_worth'] for nw in portfolio.performance.values()]

        if len(net_worths) > 1:
            return net_worths[-1] / net_worths[-min(len(net_worths), self._window_size + 1)] - 1.0
        else:
            return 0.0