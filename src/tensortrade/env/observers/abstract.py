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

import numpy as np

from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed
from tensortrade.env.mixins.scheme import SchemeMixin

if typing.TYPE_CHECKING:
    from typing import TypeAlias

    from gymnasium import Space
    from gymnasium.core import ObsType

class AbstractObserver(SchemeMixin, Component, TimeIndexed):
    """A component to generate an observation at each step of an episode.

    :param observation_dtype: The data type of the observation. Defaults to ``np.float32``.
    :type observation_dtype: TypeAlias
    :param observation_lows: The lowest value of the observation. Defaults to ``-np.inf``.
    :type observation_lows: float
    :param observation_highs: The highest value of the observation. Defaults to ``np.inf``.
    :type observation_highs: float
    """

    registered_name = "observer"

    def __init__(
            self,
            observation_dtype: TypeAlias = np.float32,
            observation_lows: float = -np.inf,
            observation_highs: float = np.inf,
    ):
        super().__init__()

        self._observation_dtype = observation_dtype
        self._observation_lows = observation_lows
        self._observation_highs = observation_highs

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space of the :class:`TradingEnv`.

        :return: The gymnasium observation space of the :class:`TradingEnv`.
        :rtype: Space
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self) -> ObsType:
        """Gets the observation at the current step of an episode.

        :return: The current observation of the environment.
        :rtype: ObsType
        """
        raise NotImplementedError()

    @abstractmethod
    def has_next(self) -> bool:
        """Tells if there is a next observation available. Actually if we still have training data available.

        :return: True if there is a next observation available, otherwise False.
        :rtype: bool
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the observer."""
        pass
