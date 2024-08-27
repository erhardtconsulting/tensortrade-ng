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
import unittest

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from tensortrade.pipeline.transformers import MutualInformationTransformer


class TestMutualInformationTransformer(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)
        data = {
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100)
        }
        dates = pd.date_range('2023-01-01', periods=100)
        self.df = pd.DataFrame(data, index=dates)
        self.transformer = MutualInformationTransformer(num_features=3, target_shift=1)

    def test_transform_dont_change_index(self):
        # Save the original index
        original_index = self.df.index

        # Test that the transform method returns a DataFrame
        result = self.transformer.transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the index has not been removed or altered
        pd.testing.assert_index_equal(result.index, original_index)

    def test_transform_returns_dataframe(self):
        # Test that the transform method returns a DataFrame
        result = self.transformer.transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_reduces_columns(self):
        # Test that the transform method reduces the DataFrame to the specified number of features
        result = self.transformer.transform(self.df)
        self.assertEqual(len(result.columns), 3)

    def test_transform_includes_close_column(self):
        # Test that the 'close' column is always included in the result
        result = self.transformer.transform(self.df)
        self.assertIn('close', result.columns)

    def test_transform_selects_top_features(self):
        # Test that the selected features are among the top by mutual information
        result = self.transformer.transform(self.df)

        # Create a new DataFrame with shifted target column
        test_df = self.df.copy()
        test_df['target_predict'] = test_df['close'].shift(-1)
        test_df.dropna(inplace=True)

        # Create X, y for mutual info regression
        X = test_df.drop(columns=['target_predict'])
        y = test_df['target_predict']

        # calculate mutual info regression
        mi_scores = mutual_info_regression(X, y)

        top_features = pd.Series(mi_scores, index=X.columns).sort_values(
            ascending=False).head(2).index.tolist()

        for feature in top_features:
            self.assertIn(feature, result.columns)

    def test_transform_handles_small_datasets(self):
        # Test the transformer with a small dataset (less rows than the target_shift)
        small_df = self.df.head(2)

        # With insufficient data, it should raise a ValueError
        with self.assertRaises(ValueError):
            self.transformer.transform(small_df)

    def test_transform_handles_custom_target_column(self):
        # Test the transformer with a custom target column
        transformer = MutualInformationTransformer(num_features=3, target_column='open', target_shift=1)
        result = transformer.transform(self.df)
        self.assertIn('open', result.columns)