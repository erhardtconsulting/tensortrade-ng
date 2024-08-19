import unittest
from unittest import TestCase
from unittest.mock import patch, Mock

import numpy as np
from numpy.ma.testutils import assert_equal

from tensortrade.env import create, actions
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
        exchange = Exchange('dummy', service=execute_order)(self.prices_stream)

        # prepare wallets
        self.usdt_wallet = Wallet(exchange, 1000 * USDT)
        self.btc_wallet = Wallet(exchange, 0 * BTC)

        # prepare portfolio
        self.portfolio = Portfolio(USDT, [
            self.usdt_wallet,
            self.btc_wallet
        ])



    def test_reward(self):
        simple_profit = SimpleProfit()

        env = create(
            feed=DataFeed([self.prices_stream]),
            portfolio=self.portfolio,
            action_scheme=actions.BSH(cash=self.usdt_wallet, asset=self.btc_wallet),
            reward_scheme=simple_profit
        )

        env.step(0)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertEqual(reward, 0.0)

        env.step(0)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertEqual(reward, 0.0, 4)

        env.step(1)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertAlmostEqual(reward, -0.0483, 4)

        env.step(1)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertAlmostEqual(reward, 0.0952, 4)

        env.step(1)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertAlmostEqual(reward, 0.0435, 4)

        env.step(0)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertAlmostEqual(reward, -0.003, 4)

        env.step(0)
        reward = simple_profit.reward(self.portfolio)
        TestCase().assertEqual(reward, 0.0)