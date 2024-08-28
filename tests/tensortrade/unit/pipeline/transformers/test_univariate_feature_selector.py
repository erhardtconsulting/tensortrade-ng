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

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from tensortrade.pipeline.transformers import UnivariateFeatureSelectionTransformer


class TestUnivariateFeatureSelectionTransformer(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for regression
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        self.df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
        self.df_reg['close'] = y_reg

        # Create a sample DataFrame for classification
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0,
                                           random_state=42)
        self.df_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(10)])
        self.df_clf['close'] = y_clf

        # Initialize transformers
        self.transformer_reg = UnivariateFeatureSelectionTransformer(num_features=5, problem_type='regression')
        self.transformer_clf = UnivariateFeatureSelectionTransformer(num_features=5, problem_type='classification')

    def test_transform_regression_returns_dataframe(self):
        # Test that the transform method for regression returns a DataFrame
        result = self.transformer_reg.transform(self.df_reg)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_classification_returns_dataframe(self):
        # Test that the transform method for classification returns a DataFrame
        result = self.transformer_clf.transform(self.df_clf)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_regression_selects_correct_number_of_features(self):
        # Test that the correct number of features is selected for regression
        result = self.transformer_reg.transform(self.df_reg)
        self.assertEqual(len(result.columns), 5)

    def test_transform_classification_selects_correct_number_of_features(self):
        # Test that the correct number of features is selected for classification
        result = self.transformer_clf.transform(self.df_clf)
        self.assertEqual(len(result.columns), 5)

    def test_transform_includes_target_column_regression(self):
        # Test that the 'close' column is always included in the regression results
        result = self.transformer_reg.transform(self.df_reg)
        self.assertIn('close', result.columns)

    def test_transform_includes_target_column_classification(self):
        # Test that the 'close' column is always included in the classification results
        result = self.transformer_clf.transform(self.df_clf)
        self.assertIn('close', result.columns)

    def test_transform_preserves_index_regression(self):
        # Save the original index for regression
        original_index = self.df_reg.index

        # Transform the DataFrame
        result = self.transformer_reg.transform(self.df_reg)

        # Check that the index has not been altered or removed
        pd.testing.assert_index_equal(result.index, original_index, "The index has been altered or removed.")

    def test_transform_preserves_index_classification(self):
        # Save the original index for classification
        original_index = self.df_clf.index

        # Transform the DataFrame
        result = self.transformer_clf.transform(self.df_clf)

        # Check that the index has not been altered or removed
        pd.testing.assert_index_equal(result.index, original_index, "The index has been altered or removed.")

    def test_transform_raises_error_on_invalid_problem_type(self):
        # Test that the transformer raises an error on an invalid problem type
        with self.assertRaises(ValueError):
            transformer_invalid = UnivariateFeatureSelectionTransformer(num_features=5, problem_type='invalid')
            transformer_invalid.transform(self.df_reg)