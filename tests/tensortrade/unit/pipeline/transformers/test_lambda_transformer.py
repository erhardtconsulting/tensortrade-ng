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

from tensortrade.pipeline.transformers import LambdaTransformer


class TestLambdaTransformer(unittest.TestCase):
    def setUp(self):
        # Initial data for the tests
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

    def test_lambda_transformer_addition(self):
        transformer = LambdaTransformer(func=lambda df: df + 1)

        transformed_df = transformer.transform(self.data)

        expected_df = self.data + 1

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_lambda_transformer_multiplication(self):
        transformer = LambdaTransformer(func=lambda df: df * 2)

        transformed_df = transformer.transform(self.data)

        expected_df = self.data * 2

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_lambda_transformer_custom_function(self):
        # Custom function that adds a new column
        transformer = LambdaTransformer(func=lambda df: df.assign(C=df['A'] + df['B']))

        transformed_df = transformer.transform(self.data)

        expected_df = self.data.copy()
        expected_df['C'] = expected_df['A'] + expected_df['B']

        pd.testing.assert_frame_equal(transformed_df, expected_df)
