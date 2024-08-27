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

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from tensortrade.pipeline.transformers.abstract import AbstractTransformer

if typing.TYPE_CHECKING:
    from typing import Optional
    from pandas import DataFrame

class CatBoostFeatureImportanceTransformer(AbstractTransformer):
    """Transformer for selecting top features based on feature importance with a target variable, calculated by
    CatBoostRegressor.

    :param num_features:  The number of top features to select. (Default = 20)
    :type num_features: int
    :param seed: The seed used for the feature importance score regression. (Default = 42)
    :type seed: int
    :param iterations: CatBoostRegressor iterations. Should be at minimum 5 to 10 times the number of features. (Default = 1000)
    :type iterations: int
    :param target_column: The name of the target column on which the mutual information score should be calculated. (Default = 'close')
    :type target_column: str
    :param target_shift: The number of periods to shift the target column to create the prediction target. (Default = 3)
    :type target_shift: int
    :param task_type: The type of the CatBoostRegressor task, can be CPU or GPU. (Default = 'CPU')
    :type task_type: str
    :param learning_rate: Learning rate used for the CatBoostRegressor. If None it is chosen dynamical by CatBoost.
    :type learning_rate: float
    :param max_depth: Max depth used for the CatBoostRegressor. (Default = 8)
    :type max_depth: int
    """
    def __init__(self,
                 num_features: int = 20,
                 seed: int = 42,
                 *,
                 iterations: int = 1000,
                 target_column: str = 'close',
                 target_shift: int = 3,
                 task_type: str = 'CPU',
                 learning_rate: Optional[float] = None,
                 max_depth: int = 8):
        self.num_features = num_features
        self.seed = seed
        self.iterations = iterations
        self.target_column = target_column
        self.target_shift = target_shift
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def transform(self, df: DataFrame) -> DataFrame:
        """Transforms the input DataFrame by selecting the top features based on feature importance score with the
        target variable, calculated with CatBoostRegressor

        :param df: The input DataFrame containing the features and target column.
        :type df: DataFrame
        :return: A DataFrame reduced to the top features based on feature importance scores.
        :rtype: DataFrame
        """
        # Create a new DataFrame with shifted target column
        test_df = df.copy()
        test_df['target_predict'] = test_df[self.target_column].shift(-self.target_shift)
        test_df.dropna(inplace=True)

        # Create X and y for training
        X = test_df.drop(columns=['target_predict'])
        y = test_df['target_predict']

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.33,
            shuffle=False,
            random_state=self.seed
        )

        # Create catboost regressor
        model = CatBoostRegressor(iterations=self.iterations,
                                  learning_rate=self.learning_rate,
                                  task_type=self.task_type,
                                  max_depth=self.max_depth,
                                  random_seed=self.seed,
                                  loss_function='RMSE',
                                  verbose=0)

        # Create train and eval pools
        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_test, y_test)

        # Train model
        model.fit(train_pool, eval_set=eval_pool)

        # Get feature importance score
        feature_importances = model.get_feature_importance(train_pool)

        # create series with importance scores
        importance_series = pd.Series(feature_importances, index=X_train.columns)
        importance_series = importance_series.sort_values(ascending=False)

        # get list of top features
        top_features = importance_series.head(self.num_features).index

        # Ensure 'self.target_column' is always in the top features and never removed
        if self.target_column not in top_features:
            top_features.insert(0, self.target_column)
            top_features.pop()

        # return only features in top feature list
        return df[top_features]
