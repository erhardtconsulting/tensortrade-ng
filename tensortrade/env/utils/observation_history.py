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
from collections import OrderedDict

import numpy as np

if typing.TYPE_CHECKING:
    from typing import Dict

class ObservationHistory:
    """Stores observations from a given episode of the environment.

    Parameters
    ----------
    window_size : int
        The amount of observations to keep stored before discarding them.

    Attributes
    ----------
    window_size : int
        The amount of observations to keep stored before discarding them.
    rows : pd.DataFrame
        The rows of observations that are used as the environment observation
        at each step of an episode.

    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.rows = OrderedDict()
        self.index = 0

    def push(self, row: Dict) -> None:
        """Stores an observation.

        Parameters
        ----------
        row : Dict
            The new observation to store.
        """
        self.rows[self.index] = row
        self.index += 1
        if len(self.rows.keys()) > self.window_size:
            del self.rows[list(self.rows.keys())[0]]

    def observe(self) -> np.array:
        """Gets the observation at a given step in an episode

        Returns
        -------
        `np.array`
            The current observation of the environment.
        """
        rows = self.rows.copy()

        if len(rows) < self.window_size:
            size = self.window_size - len(rows)
            padding = np.zeros((size, len(rows[list(rows.keys())[0]])))
            r = np.array([list(inner_dict.values()) for inner_dict in rows.values()])
            rows = np.concatenate((padding, r))

        if isinstance(rows, OrderedDict):
            rows = np.array([list(inner_dict.values()) for inner_dict in rows.values()])

        rows = np.nan_to_num(rows)

        return rows

    def reset(self) -> None:
        """Resets the observation history"""
        self.rows = OrderedDict()
        self.index = 0