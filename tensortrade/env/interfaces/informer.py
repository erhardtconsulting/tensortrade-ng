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
    from typing import Dict, Any

    from tensortrade.env.interfaces import TradingEnv


class AbstractInformer(Component, TimeIndexed):
    """A component to provide information at each step of an episode.
    """

    registered_name = "abstract_informer"

    @abstractmethod
    def info(self, env: TradingEnv) -> Dict[str, Any]:
        """Provides information at a given step of an episode.

        Parameters
        ----------
        env: 'TradingEnv'
            The trading environment.

        Returns
        -------
        dict:
            A dictionary of information about the portfolio and net worth.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the informer."""
        pass
