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

from tensortrade.pipeline.transformers import CatBoostRFECVTransformer


class TestCatBoostRFECVTransformer(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with some features
        np.random.seed(42)
        data = {
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'feature_3': np.random.rand(100) * 0.5 + np.random.rand(100) * 0.5,  # slightly correlated with target
            'feature_4': np.linspace(1, 100, 100) + np.random.rand(100) * 10,   # strongly correlated with target
            'feature_5': np.random.rand(100) * 0.1,  # random noise
            'close': np.linspace(1, 100, 100) + np.random.rand(100) * 5  # target variable
        }
        dates = pd.date_range('2023-01-01', periods=100)
        self.df = pd.DataFrame(data, index=dates)
        self.transformer = CatBoostRFECVTransformer(num_features=3, seed=42, iterations=200)

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
        # Test that the transform method reduces the DataFrame to the specified number of top features
        result = self.transformer.transform(self.df)
        self.assertLessEqual(len(result.columns), self.transformer.num_features)

    def test_transform_selects_most_important_features(self):
        # Test that the most important features according to RFECV are selected
        result = self.transformer.transform(self.df)

        # We expect 'feature_4' to be one of the top features since it's strongly correlated with 'close'
        self.assertIn('feature_4', result.columns)
        # 'feature_3' is likely to be important too, while 'feature_5' should be less relevant
        self.assertNotIn('feature_5', result.columns)

    def test_transform_no_columns_dropped(self):
        # Test the transformer when num_features is larger than available features
        transformer = CatBoostRFECVTransformer(num_features=10, seed=42, iterations=200)
        result = transformer.transform(self.df)
        pd.testing.assert_frame_equal(result, self.df, check_like=True)

    def test_transform_handles_custom_target_column(self):
        # Test the transformer with a custom target column
        self.df['custom_target'] = self.df['close'] * 1.5 + np.random.rand(100) * 5
        transformer = CatBoostRFECVTransformer(num_features=3, target_column='custom_target', seed=42, iterations=200)
        result = transformer.transform(self.df)
        self.assertIn('feature_4', result.columns)
