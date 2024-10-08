import pandas as pd
import pytest
import ta

from tensortrade.env import TradingEnv
from tensortrade.env.actions import ManagedRiskOrders
from tensortrade.env.observers import WindowObserver
from tensortrade.env.rewards import SimpleProfit

from tensortrade.feed import DataFeed, Stream, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from tests.tensortrade.unit.utils import get_path


@pytest.fixture
def portfolio():

    df1 = pd.read_csv(get_path("../../data/input/bitfinex_(BTC,ETH)USD_d.csv")).tail(100)
    df1 = df1.rename({"Unnamed: 0": "date"}, axis=1)
    df1 = df1.set_index("date")

    df2 = pd.read_csv(get_path("../../data/input/bitstamp_(BTC,ETH,LTC)USD_d.csv")).tail(100)
    df2 = df2.rename({"Unnamed: 0": "date"}, axis=1)
    df2 = df2.set_index("date")

    ex1 = Exchange("bitfinex", service=execute_order)(
        Stream.source(list(df1['BTC:close']), dtype="float").rename("USD-BTC"),
        Stream.source(list(df1['ETH:close']), dtype="float").rename("USD-ETH")
    )

    ex2 = Exchange("binance", service=execute_order)(
        Stream.source(list(df2['BTC:close']), dtype="float").rename("USD-BTC"),
        Stream.source(list(df2['ETH:close']), dtype="float").rename("USD-ETH"),
        Stream.source(list(df2['LTC:close']), dtype="float").rename("USD-LTC")
    )

    p = Portfolio(USD, [
        Wallet(ex1, 10000 * USD),
        Wallet(ex1, 10 * BTC),
        Wallet(ex1, 5 * ETH),
        Wallet(ex2, 1000 * USD),
        Wallet(ex2, 5 * BTC),
        Wallet(ex2, 20 * ETH),
        Wallet(ex2, 3 * LTC),
    ])
    return p


def test_runs_with_external_feed_only(portfolio):
    df = pd.read_csv(get_path("../../data/input/bitfinex_(BTC,ETH)USD_d.csv")).tail(100)
    df = df.rename({"Unnamed: 0": "date"}, axis=1)
    df = df.set_index("date")

    bitfinex_btc = df.loc[:, [name.startswith("BTC") for name in df.columns]]
    bitfinex_eth = df.loc[:, [name.startswith("ETH") for name in df.columns]]

    ta.add_all_ta_features(
        bitfinex_btc,
        colprefix="BTC:",
        **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )
    ta.add_all_ta_features(
        bitfinex_eth,
        colprefix="ETH:",
        **{k: "ETH:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )

    streams = []
    with NameSpace("bitfinex"):
        for name in bitfinex_btc.columns:
            streams += [Stream.source(list(bitfinex_btc[name]), dtype="float").rename(name)]
        for name in bitfinex_eth.columns:
            streams += [Stream.source(list(bitfinex_eth[name]), dtype="float").rename(name)]

    feed = DataFeed([
        Stream.group(streams).rename('features')
    ])

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = TradingEnv(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=WindowObserver(window_size=50),
        feed=feed
    )

    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)

    assert obs.shape[0] == 50


def test_runs_with_random_start(portfolio):
    df = pd.read_csv(get_path("../../data/input/bitfinex_(BTC,ETH)USD_d.csv")).tail(100)
    df = df.rename({"Unnamed: 0": "date"}, axis=1)
    df = df.set_index("date")

    bitfinex_btc = df.loc[:, [name.startswith("BTC") for name in df.columns]]
    bitfinex_eth = df.loc[:, [name.startswith("ETH") for name in df.columns]]

    ta.add_all_ta_features(
        bitfinex_btc,
        colprefix="BTC:",
        **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )
    ta.add_all_ta_features(
        bitfinex_eth,
        colprefix="ETH:",
        **{k: "ETH:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
    )

    streams = []
    with NameSpace("bitfinex"):
        for name in bitfinex_btc.columns:
            streams += [Stream.source(list(bitfinex_btc[name]), dtype="float").rename(name)]
        for name in bitfinex_eth.columns:
            streams += [Stream.source(list(bitfinex_eth[name]), dtype="float").rename(name)]

    feed = DataFeed([
        Stream.group(streams).rename('features')
    ])

    action_scheme = ManagedRiskOrders()
    reward_scheme = SimpleProfit()

    env = TradingEnv(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=WindowObserver(window_size=50),
        feed=feed,
        random_start_pct=0.10, # Randomly start within the first 10% of the sample
    )

    terminated = False
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()
        obs, _, terminated, _, _ = env.step(action)

    assert obs.shape[0] == 50
