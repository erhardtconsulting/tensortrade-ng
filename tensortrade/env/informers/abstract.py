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
from tensortrade.env.mixins.scheme import SchemeMixin

if typing.TYPE_CHECKING:
    from typing import Dict, Any


class AbstractInformer(SchemeMixin, Component, TimeIndexed):
    """A component to provide information at each step of an episode.
    """

    registered_name = "informer"

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Provides information (the info dict) at a given step of an episode.

        :return: A dictionary of information about the portfolio and net worth.
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError()

    def reset(self):
        """Performs a reset on the informer scheme. Will be called when :class:`TradingEnv` is reset."""
        pass
