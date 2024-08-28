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

from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from tensortrade.pipeline.transformers.abstract import AbstractTransformer

if typing.TYPE_CHECKING:
    from pandas import DataFrame


class UnivariateFeatureSelectionTransformer(AbstractTransformer):
    """Transformer that does univariate feature selection.

    It removes the features to num_features either by regression or classification.

    :params num_features: The number of features that should be returned. (Default = 20)
    :type num_features: int
    :param target_column: The name of the target column on which the mutual information score should be calculated. (Default = 'close')
    :type target_column: str
    :param target_shift: The number of periods to shift the target column to create the prediction target. (Default = 3)
    :type target_shift: int
    :param problem_type: The problem type to solve, either classification or regression. (Default = 'regression')
    :type problem_type: str
    """
    def __init__(self,
                 num_features: int = 20,
                 *,
                 target_column: str = 'close',
                 target_shift: int = 3,
                 problem_type: str = 'regression'):
        self.num_features = num_features
        self.target_column = target_column
        self.target_shift = target_shift
        self.problem_type = problem_type

    def transform(self, df: DataFrame) -> DataFrame:
        """Reduces features by univariate feature selection.

        :param df: The input DataFrame containing the features and target column.
        :type df: DataFrame
        :return: A DataFrame reduced to the top features based on univariate feature selection.
        :rtype: DataFrame
        :raises ValueError: If the problem_type is not 'regression' or 'classification'.
        """
        if self.problem_type == 'classification':
            # use classification
            selector = SelectKBest(score_func=f_classif, k=self.num_features)
        elif self.problem_type == 'regression':
            # use regression
            selector = SelectKBest(score_func=f_regression, k=self.num_features)
        else:
            raise ValueError('problem_type must be "classification" or "regression"')

        # Create a new DataFrame with shifted target column
        test_df = df.copy()
        test_df['target_predict'] = test_df[self.target_column].shift(-self.target_shift)
        test_df.dropna(inplace=True)

        # Create X and y for training
        X = test_df.drop(columns=['target_predict'])
        y = test_df['target_predict']

        # fit and transform data
        selector.fit_transform(X, y)

        # take only selected features
        selected_features = df.columns[selector.get_support(indices=True)].tolist()

        if self.target_column not in selected_features:
            selected_features.insert(0, self.target_column)
            selected_features.pop()

        # reduce dataframe
        df_reduced = df[selected_features]

        return df_reduced