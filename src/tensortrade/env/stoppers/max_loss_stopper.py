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

from tensortrade.env.interfaces import AbstractStopper

if typing.TYPE_CHECKING:
    from tensortrade.env.interfaces import TradingEnv

class MaxLossStopper(AbstractStopper):
    """A stopper that stops an episode if the portfolio has lost a particular
    percentage of its wealth.

    Parameters
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Attributes
    ----------
    max_allowed_loss : float
        The maximum percentage of initial funds that is willing to
        be lost before stopping the episode.

    Notes
    -----
    This stopper also stops if it has reached the end of the observation feed.
    """

    def __init__(self, max_allowed_loss: float):
        super().__init__()
        self.max_allowed_loss = max_allowed_loss

    def stop(self, env: TradingEnv) -> bool:
        c1 = env.action_scheme.portfolio.profit_loss > self.max_allowed_loss
        c2 = not env.observer.has_next()
        return c1 or c2
