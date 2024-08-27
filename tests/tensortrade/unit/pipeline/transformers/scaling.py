import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensortrade.pipeline.transformers import ScalingTransformer


class TestScalingTransformer(unittest.TestCase):
    def setUp(self):
        # initial data
        self.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        self.new_data = pd.DataFrame({
            'A': [4, 5],
            'B': [7, 8]
        })

    def test_standard_scaling(self):
        transformer = ScalingTransformer(method='standard')

        # transform data
        transformed_df = transformer.transform(self.data)

        # do transformation manually
        scaler = StandardScaler()
        expected_df = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

        # check if frames are equal
        pd.testing.assert_frame_equal(transformed_df, expected_df)

        # transform new data
        new_transformed_df = transformer.transform(self.new_data)

        # transform new data manually
        expected_new_df = pd.DataFrame(scaler.transform(self.new_data), columns=self.new_data.columns)

        # check if frames are equal
        pd.testing.assert_frame_equal(new_transformed_df, expected_new_df)

    def test_minmax_scaling(self):
        transformer = ScalingTransformer(method='minmax')

        # transform data
        transformed_df = transformer.transform(self.data)

        # do transformation manually
        scaler = MinMaxScaler()
        expected_df = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

        # check if frames are equal
        pd.testing.assert_frame_equal(transformed_df, expected_df)

        # transform new data
        new_transformed_df = transformer.transform(self.new_data)

        # transform new data manually
        expected_new_df = pd.DataFrame(scaler.transform(self.new_data), columns=self.new_data.columns)

        # check if frames are equal
        pd.testing.assert_frame_equal(new_transformed_df, expected_new_df)

    def test_invalid_method_error(self):
        # test invalid method
        with self.assertRaises(ValueError):
            ScalingTransformer(method='invalid_method')