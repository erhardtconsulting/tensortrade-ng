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
import pandas as pd

from sklearn.feature_selection import mutual_info_regression

from tensortrade.pipeline.transformers.abstract import AbstractTransformer

if typing.TYPE_CHECKING:
    from pandas import DataFrame

class MutualInformationTransformer(AbstractTransformer):
    """Transformer for selecting top features based on mutual information with a target variable.

    :param num_features:  The number of top features to select based on mutual information scores. (Default = 20)
    :type num_features: int
    :param seed: The seed used for the mutual information regression. (Default = 42)
    :type seed: int
    :param target_column: The name of the target column on which the mutual information score should be calculated. (Default = 'close')
    :type target_column: str
    :param target_shift: The number of periods to shift the target column to create the prediction target. (Default = 3)
    :type target_shift: int
    :param n_jobs: The number of parallel jobs to run. If -1, all processors are used. (Default = -1)
    :type n_jobs: int
    """
    def __init__(self,
                 num_features: int = 20,
                 seed: int = 42,
                 *,
                 target_column: str = 'close',
                 target_shift: int = 3,
                 n_jobs: int = -1
                 ):
        self.num_features = num_features
        self.seed = seed
        self.target_column = target_column
        self.target_shift = target_shift
        self.n_jobs = n_jobs

    def transform(self, df: DataFrame) -> DataFrame:
        """Transforms the input DataFrame by selecting the top features based on mutual information with the target variable.

        :param df: The input DataFrame containing the features and target column.
        :type df: DataFrame
        :return: A DataFrame reduced to the top features based on mutual information scores.
        :rtype: DataFrame
        :raises ValueError: If the number of values in the DataFrame is less than 5 after shifting.
        """
        # Create a new DataFrame with shifted target column
        test_df = df.copy()
        test_df['target_predict'] = test_df[self.target_column].shift(-self.target_shift)
        test_df.dropna(inplace=True)

        # Check if we have enough data
        if len(test_df.values) < 5:
            raise ValueError("DataFrame must have at least 5 columns after shifting.")

        # Create X, y for mutual info regression
        X = test_df.drop(columns=['target_predict'])
        y = test_df['target_predict']

        # Calculate the mutual info regression
        mi_scores = mutual_info_regression(X, y, random_state=self.seed, n_jobs=self.n_jobs)

        # Sort features by mutual info score
        mi_scores_series = pd.Series(mi_scores, index=X.columns)
        mi_scores_series = mi_scores_series.sort_values(ascending=False)

        # Select only the top 'self.num_features' with the best score
        top_features = mi_scores_series.head(self.num_features).index.tolist()

        # Ensure 'self.target_column' is always in the top features and never removed
        if self.target_column not in top_features:
            top_features.insert(0, self.target_column)
            top_features.pop()

        # Reduce the DataFrame to top features
        reduced_df = df[top_features]

        return reduced_df