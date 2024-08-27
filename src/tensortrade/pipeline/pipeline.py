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

if typing.TYPE_CHECKING:
    from typing import List
    from pandas import DataFrame

    from tensortrade.pipeline.transformers.abstract import AbstractTransformer


class DataPipeline:
    """
    Initializes the DataPipeline with a list of transformers.

    :param transformers: A list of transformer instances.
    :type transformers: List[AbstractTransformer]
    """
    def __init__(self, transformers: List[AbstractTransformer]):
        self.transformers = transformers

    def transform(self, df: DataFrame) -> DataFrame:
        """Applies each transformer in sequence to the DataFrame.

        :param df: The input DataFrame.
        :type df: DataFrame
        :return: The transformed DataFrame after applying all transformers.
        :rtype: DataFrame
        """
        for transformer in self.transformers:
            df = transformer.transform(df)

        return df
