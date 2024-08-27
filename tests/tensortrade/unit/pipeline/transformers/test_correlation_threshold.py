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

from tensortrade.pipeline.transformers import CorrelationThresholdTransformer


class TestCorrelationThresholdTransformer(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        np.random.seed(42)
        data = {
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'feature_3': np.random.rand(100)
        }
        # Slightly correlated with feature_1
        data['feature_4'] = data['feature_1'] * 0.8 + np.random.rand(100) * 0.2
        # Stronger correlated with feature_2
        data['feature_5'] = data['feature_2'] * 0.9 + np.random.rand(100) * 0.1

        dates = pd.date_range('2023-01-01', periods=100)
        self.df = pd.DataFrame(data, index=dates)
        self.transformer = CorrelationThresholdTransformer(threshold=0.85)

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
        # Manually compute correlations to predict which columns will be dropped
        corr_matrix = self.df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]

        # Test that the transform method reduces the DataFrame by removing highly correlated features
        result = self.transformer.transform(self.df)
        self.assertEqual(len(result.columns), len(self.df.columns) - len(to_drop))

    def test_transform_drops_highly_correlated_features(self):
        # Ensure that highly correlated features are indeed dropped
        result = self.transformer.transform(self.df)
        self.assertEqual(3, len(result.columns))

    def test_transform_no_columns_dropped(self):
        # Test the transformer when no columns should be dropped (i.e., lower threshold)
        transformer = CorrelationThresholdTransformer(threshold=1.0)
        result = transformer.transform(self.df)
        self.assertEqual(result.shape[1], self.df.shape[1])

    def test_transform_all_columns_dropped(self):
        # Test the transformer when all columns are highly correlated (i.e., very low threshold)
        transformer = CorrelationThresholdTransformer(threshold=0.0)
        result = transformer.transform(self.df)
        self.assertEqual(result.shape[1], 1)  # Only one column should remain