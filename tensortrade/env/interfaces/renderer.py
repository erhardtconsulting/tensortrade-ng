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
from datetime import datetime

import pandas as pd

from tensortrade.core import Component

if typing.TYPE_CHECKING:
    from collections import OrderedDict


class AbstractRenderer(Component):
    """A component for rendering a view of the environment at each step of
    an episode."""

    registered_name = "abstract_renderer"

    def __init__(self):
        super().__init__()
        self._max_episodes = None
        self._max_steps = None

    @abstractmethod
    def render(self,
               episode: int,
               max_episodes: int,
               step: int,
               max_steps: int,
               price_history: pd.DataFrame,
               net_worth: pd.Series,
               performance: pd.DataFrame,
               trades: OrderedDict) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        """Saves the rendered view of the environment."""
        pass

    def reset(self) -> None:
        """Resets the renderer."""
        pass

    def close(self) -> None:
        """Closes the renderer."""
        pass

    @staticmethod
    def _create_log_entry(episode: int = None,
                          max_episodes: int = None,
                          step: int = None,
                          max_steps: int = None,
                          date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Creates a log entry to be used by a renderer.

        Parameters
        ----------
        episode : int
            The current episode.
        max_episodes : int
            The maximum number of episodes that can occur.
        step : int
            The current step of the current episode.
        max_steps : int
            The maximum number of steps within an episode that can occur.
        date_format : str
            The format for logging the date.

        Returns
        -------
        str
            a log entry
        """
        log_entry = f"[{datetime.now().strftime(date_format)}]"

        if episode is not None:
            log_entry += f" Episode: {episode + 1}/{max_episodes if max_episodes else ''}"

        if step is not None:
            log_entry += f" Step: {step}/{max_steps if max_steps else ''}"

        return log_entry