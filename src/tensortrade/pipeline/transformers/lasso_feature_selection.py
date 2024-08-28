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

from sklearn.linear_model import Lasso

from tensortrade.pipeline.transformers.abstract import AbstractTransformer

if typing.TYPE_CHECKING:
    from pandas import DataFrame


class LassoFeatureSelectionTransformer(AbstractTransformer):
    """Transformer that uses Lasso L1 regularization to select the most important features.

    :params num_features: The number of features that should be returned. (Default = 20)
    :type num_features: int
    :param target_column: The name of the target column on which the mutual information score should be calculated. (Default = 'close')
    :type target_column: str
    :param target_shift: The number of periods to shift the target column to create the prediction target. (Default = 3)
    :type target_shift: int
    :param alpha: The alpha (strength of regularization) of lasso. (Default = 0.01)
    :type alpha: float
    :param max_iterations: The max_iterations of lasso. (Default = 1000)
    :type max_iterations: int
    :param seed: The seed used for lasso. (Default = 42)
    :type seed: int
    """
    def __init__(self,
                 num_features: int = 20,
                 *,
                 target_column: str = 'close',
                 target_shift: int = 3,
                 alpha: float = 0.01,
                 max_iterations: int = 1000,
                 seed: int = 42):
        self.num_features = num_features
        self.target_column = target_column
        self.target_shift = target_shift
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.seed = seed

    def transform(self, df: DataFrame) -> DataFrame:
        """Reduces features by lasso l1 regularization.

        :param df: The input DataFrame containing the features and target column.
        :type df: DataFrame
        :return: A DataFrame reduced to the top features based on lasso l1 regularization.
        :rtype: DataFrame
        """
        # Create a new DataFrame with shifted target column
        test_df = df.copy()
        test_df['target_predict'] = test_df[self.target_column].shift(-self.target_shift)
        test_df.dropna(inplace=True)

        # Create X and y for training
        X = test_df.drop(columns=['target_predict'])
        y = test_df['target_predict']

        # Create lasso and fit
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iterations, random_state=self.seed)
        lasso.fit(X, y)

        # Get coefficient
        lasso_coefficients = np.abs(lasso.coef_)

        # Sort features by coefficients
        top_features_indices = np.argsort(lasso_coefficients)[-self.num_features:]

        # Use the top features
        selected_features = X.columns[top_features_indices].tolist()

        if self.target_column not in selected_features:
            selected_features.insert(0, self.target_column)
            selected_features.pop()

        # reduce dataframe
        df_reduced = df[selected_features]

        return df_reduced