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

from tensortrade.pipeline.transformers import CorrelationAbsoluteTransformer


class TestCorrelationAbsoluteTransformer(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with some features
        np.random.seed(42)
        data = {
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100) * 0.1,  # random noise
            'close': np.linspace(1, 100, 100) + np.random.rand(100) * 10  # target variable
        }
        # Slightly correlated with feature_1
        data['feature_3'] = data['feature_1'] * 0.5 + np.random.rand(100) * 0.5
        # Stronger correlated with feature_2
        data['feature_4'] = data['feature_2'] * 0.9 + np.random.rand(100) * 0.1

        self.df = pd.DataFrame(data)
        self.transformer = CorrelationAbsoluteTransformer(num_features=3, price_column='close')

    def test_transform_returns_dataframe(self):
        # Test that the transform method returns a DataFrame
        result = self.transformer.transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_selects_correct_number_of_features(self):
        # Test that the correct number of features is selected
        result = self.transformer.transform(self.df)
        # Ensure it returns exactly num_features (or num_features - 1 + price_column)
        self.assertEqual(len(result.columns), self.transformer.num_features)

    def test_transform_includes_price_column(self):
        # Test that the 'price_column' is always included
        result = self.transformer.transform(self.df)
        self.assertIn(self.transformer.price_column, result.columns)

    def test_transform_preserves_index(self):
        # Save the original index
        original_index = self.df.index

        # Transform the DataFrame
        result = self.transformer.transform(self.df)

        # Check that the index has not been altered or removed
        pd.testing.assert_index_equal(result.index, original_index)

    def test_transform_least_correlating_features(self):
        # Perform the transformation
        result = self.transformer.transform(self.df)

        # Calculate the correlation matrix and mean correlation
        corr_matrix = self.df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        mean_corr = upper_tri.mean(axis=1).sort_values().dropna()

        # Get the least correlating features
        expected_features = mean_corr.tail(self.transformer.num_features).index.tolist()
        if 'close' not in expected_features:
            expected_features.insert(0, 'close')
            expected_features.pop()

        # Ensure the returned features match the expected features
        self.assertCountEqual(expected_features, result.columns)