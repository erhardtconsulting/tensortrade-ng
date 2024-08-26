# Copyright 2024 The TensorTrade-NG Authors.
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

from gymnasium.core import RenderFrame

from tensortrade.core import TimedIdentifiable
from tensortrade.env.mixins.scheme import SchemeMixin

if typing.TYPE_CHECKING:
    from typing import List, Optional, Union

class AbstractRenderer(SchemeMixin, TimedIdentifiable):
    def __init__(self):
        super().__init__()

        self.render_modes = ['human']

    @abstractmethod
    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the environment according to :class:`gymnasium.Env` specifications.

        :returns: A `RenderFrame` or a list of `RenderFrame` instances.
        :rtype: Optional[Union[RenderFrame, List[RenderFrame]]]
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the reward scheme."""
        pass