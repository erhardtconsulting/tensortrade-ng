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

from tensortrade.pipeline.transformers import DeleteColumnsTransformer


class TestDeleteColumnsTransformer(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=5)
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 200, 300, 400, 500]
        }, index=dates)

    def test_transform_dont_change_index(self):
        transformer = DeleteColumnsTransformer(columns=['B'])

        # Save the original index
        original_index = self.data.index

        # Test that the transform method returns a DataFrame
        result = transformer.transform(self.data)
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the index has not been removed or altered
        pd.testing.assert_index_equal(result.index, original_index)

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