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

from tensortrade.pipeline.transformers.abstract import AbstractTransformer

if typing.TYPE_CHECKING:
    from typing import Callable
    from pandas import DataFrame


class LambdaTransformer(AbstractTransformer):
    """Initializes the LambdaTransformer. This allows you to custom transform data without writing a class.

    :param func: A callable that takes a DataFrame and returns a transformed DataFrame.
    """
    def __init__(self, func: Callable[[DataFrame], DataFrame]):
        self.func = func

    def transform(self, df: DataFrame) -> DataFrame:
        """Applies the lambda function to the DataFrame.

        :param df: The input DataFrame.
        :type df: DataFrame
        :return: The transformed DataFrame.
        :rtype: DataFrame
        """
        return self.func(df)