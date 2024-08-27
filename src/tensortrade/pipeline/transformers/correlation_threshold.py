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

class CorrelationThresholdTransformer(AbstractTransformer):
    """Transformer that removes features based on a correlation threshold.

    :params threshold: The correlation threshold above which features are considered highly correlated
                       and one of them will be removed. (Default = 0.85)
    :type threshold: float
    :params price_column: The price column, that should not be removed.
    :type price_column: str
    """
    def __init__(self,
                 threshold: float = 0.85,
                 *,
                 price_column: str = 'close'):
        self.threshold = threshold
        self.price_column = price_column

    def transform(self, df: DataFrame) -> DataFrame:
        """Transforms the input DataFrame by removing features that are highly correlated.

        :param df: The input DataFrame containing the features.
        :type df: DataFrame
        :return: A DataFrame with features removed based on the correlation threshold.
        :rtype: DataFrame
        """
        # Calculate the absolute value correlation matrix
        corr_matrix = df.corr().abs()

        # Select the upper triangle of the correlation matrix (excluding the diagonal)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Identify features with a correlation higher than the threshold
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)
        ]

        # never remove price column
        if self.price_column in to_drop:
            to_drop.remove(self.price_column)

        # Drop the highly correlated features from the DataFrame
        return df.drop(columns=to_drop)