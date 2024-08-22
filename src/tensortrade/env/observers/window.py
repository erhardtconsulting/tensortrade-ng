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

import numpy as np

from gymnasium.spaces import Box

from tensortrade.env.observers.abstract import AbstractObserver
from tensortrade.env.utils import ObservationHistory

if typing.TYPE_CHECKING:
    from typing import TypeAlias

    from gymnasium.spaces import Space

class WindowObserver(AbstractObserver):
    """An observer that allows to use an observation window.

    This observer can use an observation window. It allows to show the data including the history to an environment. This
    can help the agent to see patterns over the time and use this information to do better decisions.

    .. note::
        There are other ways to add knowledge about how features change in times to an environment. You could also use
        options like frame stacking as example in Stable-Baselines3. The recommendation is to use only one way to add
        information about how features change in time.

    :param window_size: The size of the observation window.
    :type window_size: int
    :param observation_dtype: The data type of the observation. Defaults to ``np.float32``.
    :type observation_dtype: TypeAlias
    :param observation_lows: The lowest value of the observation. Defaults to ``-np.inf``.
    :type observation_lows: float
    :param observation_highs: The highest value of the observation. Defaults to ``np.inf``.
    :type observation_highs: float
    """

    registered_name = "default_observer"

    def __init__(
            self,
            window_size: int = 1,
            *,
            observation_dtype: TypeAlias = np.float32,
            observation_lows: float = -np.inf,
            observation_highs: float = np.inf,
            ) -> None:
        super().__init__(
            observation_dtype=observation_dtype,
            observation_lows=observation_lows,
            observation_highs=observation_highs
        )

        self.window_size = window_size
        self.history = ObservationHistory(window_size=window_size)

    @property
    def observation_space(self) -> Space:
        return Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, self.trading_env.feed.features_size),
            dtype=self._observation_dtype
        )

    def warmup(self) -> None:
        """Warms up the data feed.

        Actually fills the history until it's full to have a full window from beginning. It loads new data until we have
        enough to begin with a new episode.
        """
        if self.window_size > 1:
            for _ in range(self.window_size - 1):
                if self.has_next():
                    self.trading_env.feed.next()
                    self.history.push(self.trading_env.feed.state.features)

    def observe(self) -> np.array:
        """Observes the environment.

        This will add the actual state to the history and return the window of observations.

        :returns: The current observation window.
        :rtype: np.array
        """
        self.history.push(self.trading_env.feed.state.features)

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype)
        return obs

    def has_next(self) -> bool:
        """Checks if another observation can be generated.

        :returns: True if another observation can be generated.
        :rtype: bool
        """
        return self.trading_env.feed.has_next()

    def reset(self) -> None:
        """Resets the observer"""
        self.history.reset()
        self.warmup()
