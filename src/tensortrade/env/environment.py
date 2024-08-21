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
import uuid
import logging

from random import randint

import gymnasium
import numpy as np
import pandas as pd

from tensortrade.core import TimeIndexed, Clock, Component
from tensortrade.env.interfaces import AbstractObserver
from tensortrade.oms.orders import Broker

if typing.TYPE_CHECKING:
    from typing import Dict, Tuple, Any

    from tensortrade.env.actions.abstract import AbstractActionScheme
    from tensortrade.env.rewards.abstract import AbstractRewardScheme
    from tensortrade.env.informers.abstract import AbstractInformer

    from tensortrade.env.interfaces import (
        AbstractRenderer,
        AbstractStopper
    )
    from tensortrade.oms.wallets import Portfolio


class TradingEnv(gymnasium.Env, TimeIndexed):
    """A trading environment made for use with Gym-compatible reinforcement
    learning algorithms.

    Parameters
    ----------
    action_scheme : `AbstractActionScheme`
        A component for generating an action to perform at each step of the
        environment.
    reward_scheme : `RewardScheme`
        A component for computing reward after each step of the environment.
    observer : `AbstractObserver`
        A component for generating observations after each step of the
        environment.
    informer : `AbstractInformer`
        A component for providing information after each step of the
        environment.
    renderer : `AbstractRenderer`
        A component for rendering the environment.
    kwargs : keyword arguments
        Additional keyword arguments needed to create the environment.
    """

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 portfolio: Portfolio,
                 action_scheme: AbstractActionScheme,
                 reward_scheme: AbstractRewardScheme,
                 observer: AbstractObserver,
                 stopper: AbstractStopper,
                 informer: AbstractInformer,
                 renderer: AbstractRenderer,
                 min_periods: int = None,
                 max_episode_steps: int = None,
                 random_start_pct: float = 0.00,
                 **kwargs) -> None:
        super().__init__()


        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.informer = informer
        self.renderer = renderer
        self.min_periods = min_periods
        self.random_start_pct = random_start_pct
        self.render_mode = 'human'


        self._clock = Clock()
        self._broker = Broker()
        self._portfolio = portfolio

        # init portfolio
        self._portfolio.clock = self._clock

        # init action scheme
        self.action_scheme.trading_env = self
        self.reward_scheme.trading_env = self
        if self.informer is not None:
            self.informer.trading_env = self

        # set action and observation space
        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._enable_logger = kwargs.get('enable_logger', False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def broker(self) -> Broker:
        return self._broker

    @property
    def components(self) -> Dict[str, Component]:
        """The components of the environment. (`Dict[str,Component]`, read-only)"""
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "observer": self.observer,
            "stopper": self.stopper,
            "informer": self.informer,
            "renderer": self.renderer
        }

    def step(self, action: Any) -> Tuple[np.array, float, bool, bool, dict]:
        """Makes one step through the environment.

        Parameters
        ----------
        action : Any
            An action to perform on the environment.

        Returns
        -------
        `np.array`
            The observation of the environment after the action being
            performed.
        float
            The computed reward for performing the action.
        bool
            Whether or not the episode is complete.
        dict
            The information gathered after completing the step.
        """
        self.action_scheme.perform_action(action)

        obs = self.observer.observe(self)
        reward = self.reward_scheme.reward()
        terminated = self.stopper.stop(self)
        truncated = False
        info = self.informer.info()

        self.clock.increment()

        return obs, reward, terminated, truncated, info

    def reset(self, seed = None, options = None) -> Tuple[np.array, Dict[str, Any]]:
        """Resets the environment.

        Returns
        -------
        obs : `np.array`
            The first observation of the environment.
        """
        if self.random_start_pct > 0.00:
            size = len(self.observer.feed.process[-1].inputs[0].iterable)
            random_start = randint(0, int(size * self.random_start_pct))
        else:
            random_start = 0

        # reset env state
        self.episode_id = str(uuid.uuid4())
        self._clock.reset()
        self._portfolio.reset()
        self._broker.reset()

        # reset component state
        self.action_scheme.reset()
        self.observer.reset(random_start=random_start)
        self.reward_scheme.reset()
        if self.stopper is not None:
            self.stopper.reset()
        if self.informer is not None:
            self.informer.reset()
        if self.renderer is not None:
            self.renderer.reset()

        # return new observation
        obs = self.observer.observe(self)
        info = self.informer.info()

        self.clock.increment()

        return obs, info

    def render(self, **kwargs) -> None:
        """Renders the environment."""
        if self.renderer is not None:
            episode = kwargs.get('episode', None)
            max_episodes = kwargs.get('max_episodes', None)
            max_steps = kwargs.get('max_steps', None)

            price_history = None
            if len(self.observer.renderer_history) > 0:
                price_history = pd.DataFrame(self.observer.renderer_history)

            performance = pd.DataFrame.from_dict(self.action_scheme.portfolio.performance, orient='index')

            self.renderer.render(
                episode=episode,
                max_episodes=max_episodes,
                step=self.clock.step,
                max_steps=max_steps,
                price_history=price_history,
                net_worth=performance.net_worth,
                performance=performance.drop(columns=['base_symbol']),
                trades=self.action_scheme.broker.trades
            )

    def save(self) -> None:
        """Saves the rendered view of the environment."""
        self.renderer.save()

    def close(self) -> None:
        """Closes the environment."""
        self.renderer.close()