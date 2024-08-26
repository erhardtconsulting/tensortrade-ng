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

from tensortrade.env.renderers.abstract import AbstractRenderer

from datetime import datetime

if typing.TYPE_CHECKING:
    from typing import List

class TerminalRenderer(AbstractRenderer):
    def __init__(
            self,
            date_format: str = "%Y-%m-%d %H:%M:%S"
    ) -> None:
        super().__init__()

        self._date_format = date_format

    def render(self) -> None:
        msg = f'[{self._time()}] Episode: {self.trading_env.n_episode} - Step: {self.trading_env.clock.step}'

        if self.trading_env.last_state.reward is not None:
            msg += f' - Reward: {self.trading_env.last_state.reward}'

        msg += f' - Net worth: {self.trading_env.portfolio.net_worth}'

        if self.trading_env.last_state.terminated:
            msg += ' - Terminated'

        print(msg)

    def _time(self) -> str:
        return datetime.now().strftime(self._date_format)
