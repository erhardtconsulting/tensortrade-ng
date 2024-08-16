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

from tensortrade.env.interfaces import AbstractRenderer

if typing.TYPE_CHECKING:
    from collections import OrderedDict
    from typing import List

    import pandas as pd

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


class AggregateRenderer(AbstractRenderer):
    """A renderer that aggregates compatible renderers so they can all be used
    to render a view of the environment.

    Parameters
    ----------
    renderers : List[Renderer]
        A list of renderers to aggregate.

    Attributes
    ----------
    renderers : List[Renderer]
        A list of renderers to aggregate.
    """

    registered_name = "aggregate_renderer"

    def __init__(self, renderers: List[AbstractRenderer]) -> None:
        super().__init__()
        self.renderers = renderers

    def render(self,
               episode: int,
               max_episodes: int,
               step: int,
               max_steps: int,
               price_history: pd.DataFrame,
               net_worth: pd.Series,
               performance: pd.DataFrame,
               trades: OrderedDict) -> None:
        for r in self.renderers:
            r.render(episode=episode,
                     max_episodes=max_episodes,
                     step=step,
                     max_steps=max_steps,
                     price_history=price_history,
                     net_worth=net_worth,
                     performance=performance,
                     trades=trades)

    def save(self) -> None:
        for r in self.renderers:
            r.save()

    def reset(self) -> None:
        for r in self.renderers:
            r.reset()

    def close(self) -> None:
        for r in self.renderers:
            r.close()