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
from unittest import TestCase

import numpy as np

from tensortrade.env import TradingEnv
from tensortrade.env.actions import BSH
from tensortrade.env.observers import SimpleObserver

from tensortrade.env.rewards import SimpleProfit

from tensortrade.feed import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet


class TestSimpleProfit(unittest.TestCase):
    def setUp(self):
        self.price = np.array([1000.0, 1100.0, 1050.0, 1150.0, 1200.0, 1150.0, 1250.0, 1150.0, 1300.0])

        # Prepare trading instruments
        USDT = Instrument('USDT', 2, 'US Dollar Tether')
        BTC = Instrument('BTC', 6, 'Bitcoin')

        # prepare exchange
        self.prices_stream = Stream.source(self.price, dtype='float').rename('USDT/BTC')
        self.exchange = Exchange('dummy', service=execute_order)(self.prices_stream)

        # prepare wallets
        self.usdt_wallet = Wallet(self.exchange, 1000 * USDT)
        self.btc_wallet = Wallet(self.exchange, 0 * BTC)

        # prepare portfolio
        self.portfolio = Portfolio(USDT, [
            self.usdt_wallet,
            self.btc_wallet
        ])

    def test_reward(self):
        simple_profit = SimpleProfit()

        feed = DataFeed([
            Stream.group([self.prices_stream]).rename('features'),
        ])

        env = TradingEnv(
            portfolio=self.portfolio,
            feed=feed,
            action_scheme=BSH(cash=self.usdt_wallet, asset=self.btc_wallet),
            reward_scheme=simple_profit,
            observer=SimpleObserver()
        )

        # round 1
        env.reset()
        reward = simple_profit.reward()
        TestCase().assertEqual(0.0, reward)
        TestCase().assertEqual(0, env.clock.step)
        TestCase().assertEqual(0, self.portfolio.clock.step)
        TestCase().assertEqual(0, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 2
        env.step(0)
        reward = simple_profit.reward()
        TestCase().assertEqual(0.0, reward)
        TestCase().assertEqual(1, env.clock.step)
        TestCase().assertEqual(1, self.portfolio.clock.step)
        TestCase().assertEqual(1, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 3
        env.step(1)
        reward = simple_profit.reward()
        TestCase().assertAlmostEqual(-0.0483, reward, 4)
        TestCase().assertEqual(2, env.clock.step)
        TestCase().assertEqual(2, self.portfolio.clock.step)
        TestCase().assertEqual(2, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 4
        env.step(1)
        reward = simple_profit.reward()
        TestCase().assertAlmostEqual(0.0952, reward, 4)
        TestCase().assertEqual(3, env.clock.step)
        TestCase().assertEqual(3, self.portfolio.clock.step)
        TestCase().assertEqual(3, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 5
        env.step(1)
        reward = simple_profit.reward()
        TestCase().assertAlmostEqual(0.0435, reward, 4)
        TestCase().assertEqual(4, env.clock.step)
        TestCase().assertEqual(4, self.portfolio.clock.step)
        TestCase().assertEqual(4, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 6
        env.step(0)
        reward = simple_profit.reward()
        TestCase().assertAlmostEqual(-0.003, reward, 4)
        TestCase().assertEqual(5, env.clock.step)
        TestCase().assertEqual(5, self.portfolio.clock.step)
        TestCase().assertEqual(5, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 7
        env.step(0)
        reward = simple_profit.reward()
        TestCase().assertEqual(0.0, reward)
        TestCase().assertEqual(6, env.clock.step)
        TestCase().assertEqual(6, self.portfolio.clock.step)
        TestCase().assertEqual(6, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 8
        env.step(0)
        reward = simple_profit.reward()
        TestCase().assertEqual(0.0, reward)
        TestCase().assertEqual(7, env.clock.step)
        TestCase().assertEqual(7, self.portfolio.clock.step)
        TestCase().assertEqual(7, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())

        # round 9
        env.step(0)
        reward = simple_profit.reward()
        TestCase().assertEqual(0.0, reward)
        TestCase().assertEqual(8, env.clock.step)
        TestCase().assertEqual(8, self.portfolio.clock.step)
        TestCase().assertEqual(8, self.exchange.clock.step)
        TestCase().assertFalse(env.feed.has_next())

        # reset
        env.reset()
        reward = simple_profit.reward()
        TestCase().assertEqual(0.0, reward)
        TestCase().assertEqual(0, env.clock.step)
        TestCase().assertEqual(0, self.portfolio.clock.step)
        TestCase().assertEqual(0, self.exchange.clock.step)
        TestCase().assertTrue(env.feed.has_next())