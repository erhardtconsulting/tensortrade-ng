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

from tensortrade.env.interfaces import AbstractObserver
from tensortrade.env.utils import create_internal_streams, ObservationHistory
from tensortrade.feed import Stream, DataFeed

if typing.TYPE_CHECKING:
    from gymnasium.spaces import Space

    from tensortrade.env.interfaces import TradingEnv
    from tensortrade.oms.wallets import Portfolio

class DefaultObserver(AbstractObserver):
    """The TensorTrade observer that is compatible with the other `default`
    components.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to be used to create the internal data feed mechanism.
    feed : `DataFeed`
        The feed to be used to collect observations to the observation window.
    renderer_feed : `DataFeed`
        The feed to be used for giving information to the renderer.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    **kwargs : keyword arguments
        Additional keyword arguments for observer creation.

    Attributes
    ----------
    feed : `DataFeed`
        The master feed in charge of streaming the internal, external, and
        renderer data feeds.
    window_size : int
        The size of the observation window.
    min_periods : int
        The amount of steps needed to warmup the `feed`.
    history : `ObservationHistory`
        The observation history.
    renderer_history : `List[dict]`
        The history of the renderer data feed.
    """

    registered_name = "default_observer"

    def __init__(self,
                 portfolio: Portfolio,
                 feed: DataFeed = None,
                 renderer_feed: DataFeed = None,
                 window_size: int = 1,
                 min_periods: int = None,
                 **kwargs) -> None:
        internal_group = Stream.group(create_internal_streams(portfolio)).rename("internal")
        external_group = Stream.group(feed.inputs).rename("external")

        if renderer_feed:
            renderer_group = Stream.group(renderer_feed.inputs).rename("renderer")

            self.feed = DataFeed([
                internal_group,
                external_group,
                renderer_group
            ])
        else:
            self.feed = DataFeed([
                internal_group,
                external_group
            ])

        self.window_size = window_size
        self.min_periods = min_periods

        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.inf)
        self._observation_highs = kwargs.get('observation_highs', np.inf)

        self.history = ObservationHistory(window_size=window_size)

        initial_obs = self.feed.next()["external"]
        n_features = len(initial_obs.keys())

        self._observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )

        self.feed = self.feed.attach(portfolio)

        self.renderer_history = []

        self.feed.reset()
        self.warmup()

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def warmup(self) -> None:
        """Warms up the data feed.
        """
        if self.min_periods is not None:
            for _ in range(self.min_periods):
                if self.has_next():
                    obs_row = self.feed.next()["external"]
                    self.history.push(obs_row)

    def observe(self, env: TradingEnv) -> np.array:
        """Observes the environment.

        As a consequence of observing the `env`, a new observation is generated
        from the `feed` and stored in the observation history.

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        data = self.feed.next()

        # Save renderer information to history
        if "renderer" in data.keys():
            self.renderer_history += [data["renderer"]]

        # Push new observation to observation history
        obs_row = data["external"]
        self.history.push(obs_row)

        obs = self.history.observe()
        obs = obs.astype(self._observation_dtype)
        return obs

    def has_next(self) -> bool:
        """Checks if there is another observation to be generated.

        Returns
        -------
        bool
            Whether there is another observation to be generated.
        """
        return self.feed.has_next()

    def reset(self, random_start: int = 0) -> None:
        """Resets the observer"""
        self.renderer_history = []
        self.history.reset()
        self.feed.reset(random_start)
        self.warmup()



