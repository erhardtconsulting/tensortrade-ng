import unittest
from io import StringIO

from unittest.mock import Mock, patch

import numpy as np

from tensortrade.env import TradingEnv
from tensortrade.env.renderers import TerminalRenderer
from tensortrade.env.utils import ObsState


class TestTerminalRenderer(unittest.TestCase):

    def setUp(self):
        self.trading_env = Mock(spec=TradingEnv)

    @patch('sys.stdout', new_callable=StringIO)
    def test_render(self, mock_stdout):
        # set state
        self.trading_env.n_episode = 6
        self.trading_env.clock.step = 42
        self.trading_env.portfolio.net_worth = 1.5
        self.trading_env.last_state = ObsState(
            observation=np.array([]),
            info={},
            reward=5,
            terminated=False
        )

        renderer = TerminalRenderer()
        renderer.trading_env = self.trading_env

        renderer.render()

        self.assertEqual(f'[{renderer._time()}] Episode: 6 - Step: 42 - Reward: 5 - Net worth: 1.5\n',
                         mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=StringIO)
    def test_render_terminated(self, mock_stdout):
        # set state
        self.trading_env.n_episode = 6
        self.trading_env.clock.step = 42
        self.trading_env.portfolio.net_worth = 1.5
        self.trading_env.last_state = ObsState(
            observation=np.array([]),
            info={},
            reward=5,
            terminated=True
        )

        renderer = TerminalRenderer()
        renderer.trading_env = self.trading_env

        renderer.render()

        self.assertEqual(f'[{renderer._time()}] Episode: 6 - Step: 42 - Reward: 5 - Net worth: 1.5 - Terminated\n',
                         mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=StringIO)
    def test_render_without_reward(self, mock_stdout):
        # set state
        self.trading_env.n_episode = 8
        self.trading_env.clock.step = 99
        self.trading_env.portfolio.net_worth = 3.5
        self.trading_env.last_state = ObsState(
            observation=np.array([]),
            info={},
            reward=None,
            terminated=False
        )

        renderer = TerminalRenderer()
        renderer.trading_env = self.trading_env

        renderer.render()

        self.assertEqual(f'[{renderer._time()}] Episode: 8 - Step: 99 - Net worth: 3.5\n',
                         mock_stdout.getvalue())

