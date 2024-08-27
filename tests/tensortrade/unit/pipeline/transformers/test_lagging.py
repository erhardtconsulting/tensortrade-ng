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

from tensortrade.pipeline.transformers import LaggingTransformer


class TestLaggingTransformer(unittest.TestCase):
    def setUp(self):
        # initial data
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

    def test_lagging_transformer_single_lag(self):
        transformer = LaggingTransformer(lags=[1])

        # transform dataframe
        transformed_df = transformer.transform(self.data)

        # expected dataframe
        expected_df = self.data.copy()
        expected_df['A_lag_1'] = [None, 1.0, 2.0, 3.0, 4.0]
        expected_df['B_lag_1'] = [None, 10.0, 20.0, 30.0, 40.0]

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_lagging_transformer_multiple_lags(self):
        transformer = LaggingTransformer(lags=[1, 2], columns=['A'])

        # transform dataframe
        transformed_df = transformer.transform(self.data)

        # expected dataframe
        expected_df = self.data.copy()
        expected_df['A_lag_1'] = [None, 1.0, 2.0, 3.0, 4.0]
        expected_df['A_lag_2'] = [None, None, 1.0, 2.0, 3.0]

        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_lagging_transformer_no_lag(self):
        transformer = LaggingTransformer(columns=['A', 'B'], lags=[])

        transformed_df = transformer.transform(self.data)

        pd.testing.assert_frame_equal(transformed_df, self.data)
