import unittest

import pandas as pd

from tensortrade.pipeline import DataPipeline
from tensortrade.pipeline.transformers import LambdaTransformer, LaggingTransformer, DeleteColumnsTransformer


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Initial data for the tests
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

    def test_pipeline_with_lambda_and_lagging(self):
        pipeline = DataPipeline([
            LambdaTransformer(func=lambda df: df + 1),
            LaggingTransformer(columns=['A'], lags=[1])
        ])

        transformed_df = pipeline.transform(self.data)

        expected_df = self.data.copy()
        expected_df = expected_df + 1
        expected_df['A_lag_1'] = [None, 2, 3, 4, 5]
        expected_df.dropna(inplace=True)

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_pipeline_with_all_transformers(self):
        pipeline = DataPipeline([
            LambdaTransformer(func=lambda df: df + 1),
            LaggingTransformer(columns=['A'], lags=[1]),
            DeleteColumnsTransformer(columns=['B'])
        ])

        transformed_df = pipeline.transform(self.data)

        expected_df = self.data.copy()
        expected_df = expected_df + 1
        expected_df['A_lag_1'] = [None, 2, 3, 4, 5]
        expected_df = expected_df.drop(columns=['B'])
        expected_df.dropna(inplace=True)

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_pipeline_no_transformers(self):
        pipeline = DataPipeline([])

        transformed_df = pipeline.transform(self.data)

        # Expect the original DataFrame when no transformers are provided
        pd.testing.assert_frame_equal(transformed_df, self.data)
