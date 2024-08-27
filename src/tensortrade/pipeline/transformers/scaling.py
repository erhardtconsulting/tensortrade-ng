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

from pandas import DataFrame

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensortrade.pipeline.transformers.abstract import AbstractTransformer


class ScalingTransformer(AbstractTransformer):
    """This class is used for scaling data. So that it can be used for machine learning.

    :param method: The used scaler: 'standard' for :class:`StandardScaler`, 'minmax' for :class:`MinMaxScaler`
    :type method: str
    """
    def __init__(self, method: str = 'standard'):
        if method not in ['standard', 'minmax']:
            raise ValueError('Method should be either "standard" or "minmax".')

        self.method = method
        self.scaler = None

    def transform(self, df: DataFrame) -> DataFrame:
        """Scales the data.

        :param df: The dataframe to be scaled.
        :type df: DataFrame
        :return: The scaled dataframe.
        :rtype: DataFrame
        """
        if self.scaler is None:
            self._fit(df)

        # Scale data
        scaled_data = self.scaler.transform(df)
        return DataFrame(scaled_data, columns=df.columns)

    def _fit(self, df: DataFrame):
        """Setups the scaler.

        :param df: The dataframe to be scaled.
        :type df: DataFrame
        """
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()

        self.scaler.fit(df)