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
    from typing import List

    from pandas import DataFrame

class DeleteColumnsTransformer(AbstractTransformer):
    """Initializes the delete columns transformer.

    :param columns: List of columns to delete.
    :type columns: List[str]
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns

    def transform(self, df: DataFrame) -> DataFrame:
        """Deletes the defined columns of the DataFrames.

        :param df: The dataframe to delete the columns from.
        :type df: DataFrame
        :return: The dataframe with the columns deleted.
        :rtype: DataFrame
        """
        return df.drop(columns=self.columns, errors='ignore')
