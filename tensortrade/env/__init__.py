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
from typing import List

from tensortrade.env.environment import TradingEnv

if typing.TYPE_CHECKING:
    from typing import Optional, Union

    from tensortrade.env.interfaces import (
        AbstractActionScheme,
        AbstractRewardScheme,
        AbstractInformer,
        AbstractRenderer,
        AbstractStopper,
        AbstractObserver
    )
    from tensortrade.feed import DataFeed
    from tensortrade.oms.wallets import Portfolio


def create(portfolio: Portfolio,
           feed: DataFeed,
           action_scheme: AbstractActionScheme,
           reward_scheme: AbstractRewardScheme,
           informer: Optional[AbstractInformer] = None,
           observer: Optional[AbstractObserver] = None,
           renderer: Optional[Union[List[AbstractRenderer], AbstractRenderer]] = None,
           renderer_feed: Optional[DataFeed] = None,
           stopper: Optional[AbstractStopper] = None,
           window_size: int = 1,
           min_periods: int = None,
           random_start_pct: float = 0.00,
           max_allowed_loss: float = 0.5) -> TradingEnv:
    """Creates a default ``TradingEnv`` to be used by a RL agent of your choice. It allows you

    :param portfolio: Portfolio: The portfolio that the RL agent will be interacting with.
    :param feed: DataFeed: The data feed for the look back window with the ohlcv and feature data.
    :param action_scheme: AbstractActionScheme: The action scheme used by the TradingEnv.
    :param reward_scheme: AbstractRewardScheme: The reward scheme applied to the RL agent.
    :param informer: Optional[AbstractInformer]: The information logger which runs on every episode. (Default value = None)
    :param observer: Optional[AbstractObserver]: The observer which will create the observation for the RL agent. If ``None``, the default observer will be used. (Default value = None)
    :param renderer: Optional[Union[List[AbstractRenderer], AbstractRenderer]]: A renderer which will be used for rendering the environment. Like for creating charts. Will be executed when :code:`env.render()` is called. You can insert a list if you want to use more than one renderer. (Default value = None)
    :param renderer_feed: Optional[DataFeed]: An optional feed for the renderer, mostly with the actual prices used for rendering. (Default value = None)
    :param stopper: Optional[AbstractStopper]: The stopper which resets the environment on the defined circumstanced. If ``None``, the MaxLossStopper will be used which resets the environment on :code:`max_allowed_loss`. (Default value = None)
    :param window_size: int: The window size which will used by the default observer. Actually the timerange of your data that agent sees. (Default value = 1)
    :param min_periods: int: The amount of steps needed to warm up the :code:`feed`. So actually when the first episode starts. (Default value = None)
    :param random_start_pct: float: If the agent should randomly start after this percent of data. Can be used to prevent overfitting. (Default value = 0.00)
    :param max_allowed_loss: float: When using the default stopper this is max loss the agent is allowed to have before it gets reseted (Default value = 0.5)
    :rtype: TradingEnv
    :returns: A training environment you can use for training a reinforcement learning agent.
    """
    from tensortrade.env import actions, informers, observers, renderers, rewards, stoppers
    from tensortrade.env.renderers.utils import AggregateRenderer

    # set portfolio of action scheme
    action_scheme.portfolio = portfolio

    # prepare observer
    if observer is None:
        observer = observers.DefaultObserver(
            portfolio=portfolio,
            feed=feed,
            renderer_feed=renderer_feed,
            window_size=window_size,
            min_periods=min_periods
        )

    # prepare stopper
    if stopper is None:
        stopper = stoppers.MaxLossStopper(
            max_allowed_loss=max_allowed_loss
        )

    # prepare informer
    if informer is None:
        informer = informers.SimpleInformer()

    # prepare renderer
    if isinstance(renderer, List):
        renderer = AggregateRenderer(renderer)

    # create env
    return TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=stopper,
        informer=informer,
        renderer=renderer,
        min_periods=min_periods,
        random_start_pct=random_start_pct,
    )
