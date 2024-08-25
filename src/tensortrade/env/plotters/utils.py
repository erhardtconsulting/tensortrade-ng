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

import os
import typing
from datetime import datetime

from tensortrade.env.mixins.scheme import SchemeMixin
from tensortrade.env.plotters.abstract import AbstractPlotter

if typing.TYPE_CHECKING:
    from typing import List

    from tensortrade.env import TradingEnv

def create_auto_file_name(filename_prefix: str,
                           ext: str,
                           timestamp_format: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = datetime.now().strftime(timestamp_format)
    filename = filename_prefix + timestamp + '.' + ext
    return filename


def check_path(path: str, auto_create: bool = True) -> None:
    if not path or os.path.exists(path):
        return

    if auto_create:
        os.mkdir(path)
    else:
        raise OSError(f"Path '{path}' not found.")


def check_valid_format(valid_formats: list, save_format: str) -> None:
    if save_format not in valid_formats:
        raise ValueError("Acceptable formats are '{}'. Found '{}'".format("', '".join(valid_formats), save_format))


class AggregatePlotter(AbstractPlotter):
    """A renderer that aggregates compatible plotters so they can all be used
    to render a view of the environment.

    Parameters
    ----------
    renderers : List[Renderer]
        A list of plotters to aggregate.

    Attributes
    ----------
    renderers : List[Renderer]
        A list of plotters to aggregate.
    """

    registered_name = "aggregate_renderer"

    def __init__(self, renderers: List[AbstractPlotter]) -> None:
        super().__init__()
        self.renderers = renderers

    @SchemeMixin.trading_env.setter
    def trading_env(self, value: TradingEnv):
        """Sets the :class:`TradingEnv` instance.

        This setter allows for the initialization of the `_trading_env` attribute.

        :param value: The `TradingEnv` instance to be assigned to `_trading_env`.
        :type value: TradingEnv
        """
        self._trading_env = value

        for renderer in self.renderers:
            renderer.trading_env = value

    def render(self) -> None:
        for r in self.renderers:
            r.render()

    def save(self) -> None:
        for r in self.renderers:
            r.save()

    def reset(self) -> None:
        for r in self.renderers:
            r.reset()

    def close(self) -> None:
        for r in self.renderers:
            r.close()