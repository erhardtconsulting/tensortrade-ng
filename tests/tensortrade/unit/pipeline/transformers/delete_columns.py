import unittest

import pandas as pd

from tensortrade.pipeline.transformers import DeleteColumnsTransformer


class TestDeleteColumnsTransformer(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        })

    def test_delete_single_column(self):
        transformer = DeleteColumnsTransformer(columns=['B'])

        transformed_df = transformer.transform(self.data)

        expected_df = self.data.drop(columns=['B'])

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_delete_multiple_columns(self):
        transformer = DeleteColumnsTransformer(columns=['B', 'C'])

        transformed_df = transformer.transform(self.data)

        expected_df = self.data.drop(columns=['B', 'C'])

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_delete_non_existing_column(self):
        transformer = DeleteColumnsTransformer(columns=['D'])

        transformed_df = transformer.transform(self.data)

        # If the column does not exist, nothing should happen
        pd.testing.assert_frame_equal(transformed_df, self.data)