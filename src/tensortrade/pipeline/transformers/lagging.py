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
    from typing import List, Optional

    from pandas import DataFrame


class LaggingTransformer(AbstractTransformer):
    """Initialized the lagging transformer.

    :param lags: List of lags to be added.
    :type lags: List[int]
    :param columns: List of columns to lag. If none, all columns are used.
    :type columns: Optional[List[str]]
    """
    def __init__(self, lags: List[int], columns: Optional[List[str]] = None):
        self.lags = lags
        self.columns = columns

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Adds lagged features to the dataframe.

        :param df: The dataframe to add the lagged features to.
        :type df: DataFrame
        :return: The dataframe with the lags.
        :rtype: DataFrame
        """
        df_lagged = df.copy()

        if self.columns is not None:
            lag_columns = self.columns
        else:
            lag_columns = df.columns

        for column in lag_columns:
            for lag in self.lags:
                df_lagged[f'{column}_lag_{lag}'] = df_lagged[column].shift(lag)

        return df_lagged.dropna()