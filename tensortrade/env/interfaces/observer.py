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

if typing.TYPE_CHECKING:
    import numpy as np

    from gymnasium import Space

    from tensortrade.env.interfaces import TradingEnv

class AbstractObserver(Component, TimeIndexed):
    """A component to generate an observation at each step of an episode.
    """

    registered_name = "abstract_observer"

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the `TradingEnv`. (`Space`, read-only)
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self, env: TradingEnv) -> np.array:
        """Gets the observation at the current step of an episode

        Parameters
        ----------
        env: 'TradingEnv'
            The trading environment.

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, random_start: int = 0):
        """Resets the observer."""
        pass
