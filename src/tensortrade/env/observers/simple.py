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
    """A simple observer that allows to observe the data.

    This observer just returns the feature data. It is the simplest observer possible.

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
            observation_dtype: TypeAlias = np.float32,
            observation_lows: float = -np.inf,
            observation_highs: float = np.inf,
            ) -> None:
        super().__init__(
            observation_dtype=observation_dtype,
            observation_lows=observation_lows,
            observation_highs=observation_highs
        )

    @property
    def observation_space(self) -> Space:
        return Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(1, self.trading_env.feed.features_size),
            dtype=self._observation_dtype
        )

    def observe(self) -> np.array:
        """Observes the environment.

        This will return the actual state of the features.

        :returns: The current observation window.
        :rtype: np.array
        """
        obs = np.array(self.trading_env.feed.state.features.values())
        obs = obs.astype(self._observation_dtype)

        return obs

    def has_next(self) -> bool:
        """Checks if another observation can be generated.

        :returns: True if another observation can be generated.
        :rtype: bool
        """
        return self.trading_env.feed.has_next()
