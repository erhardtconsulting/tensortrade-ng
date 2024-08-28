import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

from tensortrade.pipeline.transformers import LassoFeatureSelectionTransformer


class TestLassoFeatureSelectionTransformer(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with some features
        np.random.seed(42)
        data = {
            'feature_1': np.random.rand(100),
            'feature_2': np.linspace(1, 100, 100) * 0.5 + np.random.rand(100) * 0.5,  # slightly correlated with target
            'feature_3': np.linspace(1, 100, 100) * 0.6 + np.random.rand(100) * 0.4,  # slightly correlated with target
            'feature_4': np.random.rand(100) * 0.1,  # random noise
            'close': np.linspace(1, 100, 100) + np.random.rand(100) * 10  # target variable
        }
        self.df = pd.DataFrame(data)
        self.transformer = LassoFeatureSelectionTransformer(num_features=3, alpha=0.1, max_iterations=1000, seed=42)

    def test_transform_returns_dataframe(self):
        # Test that the transform method returns a DataFrame
        result = self.transformer.transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_selects_correct_number_of_features(self):
        # Test that the correct number of features is selected
        result = self.transformer.transform(self.df)
        self.assertEqual(len(result.columns), 3)

    def test_transform_includes_target_column(self):
        # Test that the 'close' column is always included
        result = self.transformer.transform(self.df)
        self.assertIn('close', result.columns)

    def test_transform_preserves_index(self):
        # Save the original index
        original_index = self.df.index

        # Transform the DataFrame
        result = self.transformer.transform(self.df)

        # Check that the index has not been altered or removed
        pd.testing.assert_index_equal(result.index, original_index)

    def test_transform_selects_features_with_highest_coefficients(self):
        # Perform the transformation
        result = self.transformer.transform(self.df)

        # Create Lasso and fit to get the coefficients directly
        X = self.df.drop(columns=['close'])
        y = self.df['close'].shift(-self.transformer.target_shift).dropna()
        lasso = Lasso(alpha=self.transformer.alpha, max_iter=self.transformer.max_iterations,
                      random_state=self.transformer.seed)
        lasso.fit(X.iloc[:-self.transformer.target_shift], y)

        # Get coefficient
        lasso_coefficients = np.abs(lasso.coef_)

        # Sort features by coefficients and get the top ones
        top_features_indices = np.argsort(lasso_coefficients)[-self.transformer.num_features:]
        top_features = X.columns[top_features_indices].tolist()

        # Check if the selected features match the expected top features
        self.assertTrue(set(result.columns).issubset(set(top_features + ['close'])))
