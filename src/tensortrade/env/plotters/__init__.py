# Copyright 2024 The TensorTrade and TensorTrade-NG Authors.
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
import importlib.util

if importlib.util.find_spec("matplotlib"):
    from tensortrade.env.plotters.matplotlib_trading_chart import MatplotlibTradingChart

if importlib.util.find_spec("plotly"):
    from tensortrade.env.plotters.plotly_trading_chart import PlotlyTradingChart