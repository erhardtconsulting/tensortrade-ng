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
import numpy as np

from tensortrade.pipeline.transformers.abstract import AbstractTransformer

if typing.TYPE_CHECKING:
    from pandas import DataFrame

class CorrelationAbsoluteTransformer(AbstractTransformer):
    """Transformer that removes features based on their correlation.

    It creates a rank based on the correlation and removes the features with the
    highest correlation. Then it returns num_features with the least correlation.

    :params num_features: The number of features that should be returned. (Default = 20)
    :type num_features: int
    :params price_column: The price column, that should not be removed. (Default = 'close')
    :type price_column: str
    """
    def __init__(self,
                 num_features: int = 20,
                 *,
                 price_column: str = 'close'):
        self.num_features = num_features
        self.price_column = price_column

    def transform(self, df: DataFrame) -> DataFrame:
        """Transforms the input DataFrame by returning the least correlating features.

        :param df: The input DataFrame containing all features.
        :type df: DataFrame
        :return: A DataFrame with the least correlating features.
        :rtype: DataFrame
        """
        # Calculate the absolute value correlation matrix
        corr_matrix = df.corr().abs()

        # upper triangle of the correlation matrix (excluding the diagonal)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # calculate mean correlation of every feature
        mean_corr = upper_tri.mean(axis=1)

        # sort features by mean correlation and drop NaNs
        sorted_features = mean_corr.sort_values(ascending=False).dropna()

        # build list of selected features
        selected_features = sorted_features.tail(self.num_features).index.tolist()

        # never remove price column
        if self.price_column not in selected_features:
            selected_features.insert(0, self.price_column)
            selected_features.pop()

        # Drop the highly correlated features from the DataFrame
        return df[selected_features]