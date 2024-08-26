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

import random
import typing
import uuid
from warnings import warn

import gymnasium

from typing import List

from gymnasium.core import ActType, ObsType, RenderFrame

from tensortrade.core import TimeIndexed, Clock, Component
from tensortrade.env.plotters.utils import AggregatePlotter
from tensortrade.env.utils import FeedController, ObsState
from tensortrade.feed import DataFeed
from tensortrade.oms.orders import Broker

if typing.TYPE_CHECKING:
    from typing import Dict, Tuple, Any, SupportsFloat, Optional, Union

    from tensortrade.env.actions.abstract import AbstractActionScheme
    from tensortrade.env.observers.abstract import AbstractObserver
    from tensortrade.env.rewards.abstract import AbstractRewardScheme
    from tensortrade.env.renderers.abstract import AbstractRenderer
    from tensortrade.env.plotters.abstract import AbstractPlotter
    from tensortrade.env.stoppers.abstract import AbstractStopper
    from tensortrade.env.informers.abstract import AbstractInformer
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
    render_mode : str
        The chosen render mode. As example 'human'.
    plotter : `AbstractPlotter`
        A component for rendering the environment.
    """

    def __init__(self,
                 portfolio: Portfolio,
                 feed: DataFeed,
                 action_scheme: AbstractActionScheme,
                 reward_scheme: AbstractRewardScheme,
                 observer: AbstractObserver,
                 *,
                 stopper: Optional[AbstractStopper] = None,
                 informer: Optional[AbstractInformer] = None,
                 renderer: Optional[AbstractRenderer] = None,
                 render_mode: Optional[str] = None,
                 plotter: Union[Optional[AbstractPlotter], List[AbstractPlotter]] = None,
                 random_start_pct: float = 0.00
                 ) -> None:
        super().__init__()

        self.random_start_pct = random_start_pct

        # public variables
        self.agent_id: Optional[str] = None
        self.episode_id: Optional[str] = None
        self.n_episode: int = -1
        self.metadata: Dict[str, Any] = {}

        # private variables
        self._action_scheme = action_scheme
        self._reward_scheme = reward_scheme
        self._observer = observer
        self._stopper = stopper
        self._informer = informer
        self._portfolio = portfolio
        self._renderer = renderer

        # renderer can be a list of multiple plotters
        if plotter is not None and isinstance(plotter, List):
            self._plotter = AggregatePlotter(renderers=plotter)
        else:
            self._plotter = plotter

        # init portfolio
        self._portfolio.clock = self._clock

        # internal attributes
        self._broker = Broker()
        self._feed = FeedController(feed, self._portfolio)
        self._last_state: Optional[ObsState] = None

        # init components
        self._action_scheme.trading_env = self
        self._reward_scheme.trading_env = self
        self._observer.trading_env = self

        if self._renderer is not None:
            self._renderer.trading_env = self
            self.metadata['render_modes'] = self._renderer.render_modes

        if self._plotter is not None:
            self._plotter.trading_env = self

        if self._stopper is not None:
            self._stopper.trading_env = self

        if self._informer is not None:
            self._informer.trading_env = self

        # set action and observation space
        self.action_space = self._action_scheme.action_space
        self.observation_space = self._observer.observation_space

        # configure renderer
        if render_mode is not None:
            if render_mode in self.metadata['render_modes']:
                self.render_mode = render_mode
            else:
                warn(f'Render mode "{render_mode}" is not supported.', UserWarning)

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
    def feed(self) -> FeedController:
        return self._feed

    @property
    def last_state(self) -> ObsState:
        return self._last_state

    @property
    def components(self) -> Dict[str, Component]:
        """The components of the environment. (`Dict[str,Component]`, read-only)"""
        return {
            "action_scheme": self._action_scheme,
            "reward_scheme": self._reward_scheme,
            "observer": self._observer,
            "stopper": self._stopper,
            "informer": self._informer,
            "renderer": self._renderer
        }

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        .. note::
            The tuple returned contains the data according to :class:`gymnasium.Env` specifications:
                * observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation
                  due to the agent actions.
                  This could be a numpy array with the observed features at that time.
                * reward (SupportsFloat): The reward as a result of taking the action.
                * terminated (bool): Whether the agent reaches the terminal state which can be positive or negative. This
                  happens when there is no training data anymore or by the metric defined by :class:`AbstractStopper`.
                * truncated (bool): Whether the truncation condition outside the scope is satisfied. This is not used by
                  TensorTrade-NG.
                * info (Dict[str, Any]): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                  It's controlled by the :class:`AbstractInformer`.

        .. note::
            Because the internals of this method may look a bit special, hereby a little explanation:
                #. The first step is to execute the action defined by the :class:`AbstractActionScheme`. After executing,
                   we will have orders executed and therefor need to get new data.
                #. We will now use ``self.feed.next()`` to fetch the newest data with the changes (like Orders) that the
                   action has done to the environment. We begin a new state.
                #. Now we are ready to reward this new state by using the :class:`AbstractRewardScheme`.
                #. After rewarding the agent we can get a new observation and info from this new state.
                #. Last but not least we need to check if it's time to terminate this episode. This can either happen because
                   :class:`AbstractStopper` decides it, or we don't have any more data to begin a new state.

        :param action: An action provided by the agent to update the environment state.
        :type action: ActType
        :return: A :class:`gymnasium.Env` observation of the environment to learn the agent.
        :rtype: Tuple[ObsType, float, bool, bool, Dict[str, Any]]
        """

        # Execute the action decided by the agent
        self._action_scheme.perform_action(action)

        # Get new data and begin a new state
        self.clock.increment()
        self.feed.next()

        # Reward the agent, get a new observation for the next decision and add the info
        reward = self._reward_scheme.reward()
        obs, info = self._get_obs()

        # Now we decide if we need to end this episode
        if self._stopper is not None:
            terminated = self._stopper.stop()
        else:
            terminated = False

        # If we are not terminated right now, check if there is still data available
        if not terminated:
            terminated = not self.feed.has_next()

        # Save last state
        self._last_state = ObsState(
            observation=obs,
            info=info,
            reward=reward,
            terminated=terminated
        )

        if self.render_mode == 'human':
            self._renderer.render()

        return obs, reward, terminated, False, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method resets all components of the environment to it's initial state and begins with a new episode. The seed
        parameter is used to reset the PRNG of the environment. It should always be used to initialize the environment and
        after the environment is terminated or truncated. Then it returns the first observation and info according to the
        used components.

        .. note::
            The tuple returned contains the data according to :class:`gymnasium.Env` specifications:
                * observation (ObsType): The first observation of the environment. Like in ``step()``.
                * info (Dict[str, Any]): The info-dict like in ``step()``

        :return: A :class:`gymnasium.Env` initial observation.
        :rtype: Tuple[ObsType, Dict[str, Any]]
        """
        super().reset(seed=seed)

        # reset all components
        self._reset_env(seed=seed)

        # return new observation
        obs, info = self._get_obs()

        # Save last state
        self._last_state = ObsState(
            observation=obs,
            info=info
        )

        if self.render_mode == 'human':
            self._renderer.render()

        return obs, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the environment according to :class:`gymnasium.Env` specifications.

        :returns: A `RenderFrame` or a list of `RenderFrame` instances.
        :rtype: Optional[Union[RenderFrame, List[RenderFrame]]]
        """
        if self._renderer is not None:
            return self._renderer.render()

    def plot(self, **kwargs) -> None:
        """Renders the environment."""
        if self._plotter is not None:
            self._plotter.render()

    def close(self) -> None:
        """Closes the environment."""
        self._plotter.close()

    def _get_obs(self) -> Tuple[ObsType, Dict[str, Any]]:
        obs = self._observer.observe()

        if self._informer is not None:
            info = self._informer.info()
        else:
            info = {}

        return obs, info

    def _reset_env(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

        if self.random_start_pct > 0:
            random_start = random.randint(0, self.feed.features_len)
        else:
            random_start = 0

        # reset env state
        self.episode_id = str(uuid.uuid4())
        self.n_episode += 1
        self._clock.reset()
        self._portfolio.reset()
        self._broker.reset()
        self._feed.reset(random_start=random_start)

        # reset component state
        self._action_scheme.reset()
        self._observer.reset()
        self._reward_scheme.reset()

        if self._stopper is not None:
            self._stopper.reset()

        if self._informer is not None:
            self._informer.reset()

        if self._renderer is not None:
            self._renderer.reset()

        if self._plotter is not None:
            self._plotter.reset()
