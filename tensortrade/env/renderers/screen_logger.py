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

from tensortrade.env.interfaces import AbstractRenderer

if typing.TYPE_CHECKING:
    import pandas as pd

    from collections import OrderedDict

class ScreenLogger(AbstractRenderer):
    """Logs information the screen of the user.

    Parameters
    ----------
    date_format : str
        The format for logging the date.
    """

    registered_name = "screen_logger"

    DEFAULT_FORMAT: str = "[%(asctime)-15s] %(message)s"

    def __init__(self, date_format: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__()
        self._date_format = date_format

    def render(self,
               episode: int,
               max_episodes: int,
               step: int,
               max_steps: int,
               price_history: pd.DataFrame,
               net_worth: pd.Series,
               performance: pd.DataFrame,
               trades: OrderedDict) -> None:

        print(self._create_log_entry(episode, max_episodes, step, max_steps, date_format=self._date_format))
