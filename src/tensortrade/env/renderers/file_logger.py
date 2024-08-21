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

import logging
import os
import typing

from tensortrade.env.interfaces import AbstractRenderer
from tensortrade.env.renderers.utils import create_auto_file_name, check_path

if typing.TYPE_CHECKING:
    import pandas as pd

    from collections import OrderedDict


class FileLogger(AbstractRenderer):
    """Logs information to a file.

    Parameters
    ----------
    filename : str
        The file name of the log file. If omitted, a file name will be
        created automatically.
    path : str
        The path to save the log files to. None to save to same script directory.
    log_format : str
        The log entry format as per Python logging. None for default. For
        more details, refer to https://docs.python.org/3/library/logging.html
    timestamp_format : str
        The format of the timestamp of the log entry. Node for default.
    """

    registered_name = "file_logger"

    DEFAULT_LOG_FORMAT: str = '[%(asctime)-15s] %(message)s'
    DEFAULT_TIMESTAMP_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    def __init__(self,
                 filename: str = None,
                 path: str = 'log',
                 log_format: str = None,
                 timestamp_format: str = None) -> None:
        super().__init__()
        check_path(path)

        if not filename:
            filename = create_auto_file_name('log_', 'log')

        self._logger = logging.getLogger(self.id)
        self._logger.setLevel(logging.INFO)

        if path:
            filename = os.path.join(path, filename)
        handler = logging.FileHandler(filename)
        handler.setFormatter(
            logging.Formatter(
                log_format if log_format is not None else self.DEFAULT_LOG_FORMAT,
                datefmt=timestamp_format if timestamp_format is not None else self.DEFAULT_TIMESTAMP_FORMAT
            )
        )
        self._logger.addHandler(handler)

    @property
    def log_file(self) -> str:
        """The filename information is being logged to. (str, read-only)
        """
        return self._logger.handlers[0].baseFilename

    def render(self,
               episode: int,
               max_episodes: int,
               step: int,
               max_steps: int,
               price_history: pd.DataFrame,
               net_worth: pd.Series,
               performance: pd.DataFrame,
               trades: OrderedDict) -> None:

        log_entry = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._logger.info(f"{log_entry} - Performance:\n{performance}")