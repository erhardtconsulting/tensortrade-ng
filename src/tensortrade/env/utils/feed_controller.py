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
from __future__ import annotations

import typing

import pandas as pd

from dataclasses import dataclass

from warnings import warn

from pandas.core.interchange.dataframe_protocol import DataFrame

from tensortrade.core import Observable, TimeIndexed
from tensortrade.feed import Stream, DataFeed
from tensortrade.feed.core.base import IterableStream, Group, NameSpace

if typing.TYPE_CHECKING:
    from pandas import DataFrame

    from typing import Any, Dict, List, Optional

    from tensortrade.oms.wallets import Portfolio, Wallet


@dataclass
class State:
    """The actual state of the environment.

    :param features: The features at this point in time.
    :type features: Dict[str, Any]
    :param meta: The metadata at this point in time. Could be None.
    :type meta: Optional[Dict[str, Any]]
    :param meta: The portfolio data at this point in time.
    :type meta: Dict[str, Any]
    """
    features: Dict[str, Any]
    meta: Optional[Dict[str, Any]]
    portfolio: Dict[str, Any]
    step: int


class FeedController(Observable, TimeIndexed):
    """This class is responsible for controlling the global feed in a :class:`TradingEnv`.

    The :class:`FeedController` prepares the feed on initialization and provides access to the features, metadata and
    portfolio data. It has to be reset when the environment is reset and for new data ``next()`` has to be executed.

    .. note::
        The feed need to be a group of streams compiled together to a :class:`DataFeed`. It should consists of:
            * features (Group): The features shown to the environment as observation for learning. This data should
              be normalized and prepared for the environment. If it's missing a :class:`ValueError` is raised.
            * meta (Group): The metadata used by the components, like plotters or plotters. This contains data like
              raw ohlcv data. It can be omitted but this will display a warning.

    :param feed: The feed to use.
    :type feed: DataFeed
    :param portfolio: The portfolio to fetch data from.
    :type portfolio: Portfolio
    """
    def __init__(
            self,
            feed: DataFeed,
            portfolio: Portfolio
    ):
        super().__init__()

        self._meta_history: List[Dict[str, Any]] = []

        self._prepare_feed(feed, portfolio)
        self._update_data()

    @property
    def state(self) -> State:
        """Gets the state at this point in time.

        :return: The state.
        :rtype: State
        """
        return self._state

    @property
    def meta_history(self) -> DataFrame:
        """Gets the metadata history of this episode.

        :return: The metadata history.
        :rtype: DataFrame
        """
        return pd.DataFrame(self._meta_history)

    @property
    def features_len(self) -> int:
        """Gets the number of available features for training the model.

        :return: The number of features.
        :rtype: int
        """
        return self._feature_len

    @property
    def features_size(self) -> int:
        """Gets the number of features per state.

        :return: The number of features per state.
        :rtype: int
        """
        return self._feature_size

    def has_next(self) -> bool:
        """Checks if there is more data available.

        :return: Whether there is new data available.
        :rtype: bool
        """
        return self._feed.has_next()

    def next(self) -> None:
        """Get next data."""
        self._update_data()

    def reset(self, random_start: int = 0) -> None:
        """Resets the feed and gets first data"""
        self._meta_history = []
        self._feed.reset(random_start=random_start)
        self._update_data()

    def _update_data(self) -> None:
        """Updates the data variables"""
        data = self._feed.next()
        meta = data.get('meta')
        self._state = State(
            features=data.get('features'),
            meta=data.get('meta'),
            portfolio=data.get('portfolio'),
            step=self.clock.step
        )

        if meta is not None:
            self._meta_history.append(meta)

        for listener in self.listeners:
            listener.on_next(self._state)

    def _prepare_feed(self, input_feed: DataFeed, portfolio: Portfolio) -> None:
        """Prepares the feed to be used by the environment and all other components.

        .. note::
            The DataFeed is a group of streams compiled together to be used by the environment. It consists of:
                * features (Group): The features shown to the environment as observation for learning. This data should
                  be normalized and prepared for the environment.
                * meta (Group): The metadata used by the components, like plotters or plotters. This contains data like
                  raw OHLCV data.
                * portfolio (Group): This contains data about the portfolio and will be created by the environment itself.

        :param input_feed: The input feed to be used by the environment. Must contain at least a features group of streams.
        :type input_feed: DataFeed
        """
        feed = []

        # select features feed and do some checks
        try:
            features_feed = Stream.select(input_feed.inputs, lambda s: s.name is 'features' and isinstance(s, Group))
            for fs in features_feed.inputs:
                if not isinstance(fs, IterableStream):
                    raise ValueError('Environment only supports IterableStreams.')

            self._feature_len = len(features_feed.inputs[0].iterable)
            self._feature_size = len(features_feed.inputs)

            # display a warning when the user selects to many features.
            if self._feature_size > 20:
                warn('Your feature set contains more than 20 features. This may introduce noise and lead to overfitting, '
                     'potentially reducing your model\'s effectiveness. Consider reducing the number of features to improve '
                     'performance and accuracy.', UserWarning)

            feed += [features_feed]
        except AttributeError:
            raise AttributeError('Feed has no features feed.')

        # select meta feed and do checks
        try:
            meta_feed = Stream.select(input_feed.inputs, lambda s: s.name is 'meta' and isinstance(s, Group))
            for fs in features_feed.inputs:
                if not isinstance(fs, IterableStream):
                    raise ValueError('Environment only supports IterableStreams.')

            feed += [meta_feed]
        except AttributeError:
            warn('Feed has no meta feed. Therefor some components may not work.', UserWarning)

        # add portfolio
        feed += [Stream.group(self.create_portfolio_streams(portfolio)).rename('portfolio')]

        self._feed = DataFeed(feed)
        self._feed.compile()
        self.attach(portfolio)

    @staticmethod
    def create_wallet_source(wallet: Wallet, include_worth: bool = True) -> List[Stream[float]]:
        """Creates a list of streams to describe a :class:`Wallet`.

        :param wallet: The wallet to create the streams for.
        :type wallet: Wallet
        :param include_worth: Whether to include the worth of the wallet.
        :type include_worth: bool
        :return: A list of streams to describe a :class:`Wallet`.
        :rtype: List[Stream[float]]
        """
        exchange_name = wallet.exchange.name
        symbol = wallet.instrument.symbol

        streams = []

        with NameSpace(exchange_name + ":/" + symbol):
            free_balance = Stream.sensor(wallet, lambda w: w.balance.as_float(), dtype="float").rename("free")
            locked_balance = Stream.sensor(wallet, lambda w: w.locked_balance.as_float(), dtype="float").rename(
                "locked")
            total_balance = Stream.sensor(wallet, lambda w: w.total_balance.as_float(), dtype="float").rename("total")

            streams += [free_balance, locked_balance, total_balance]

            if include_worth:
                price = Stream.select(wallet.exchange.streams(), lambda node: node.name.endswith(symbol))
                worth = price.mul(total_balance).rename('worth')
                streams += [worth]

        return streams

    @staticmethod
    def create_portfolio_streams(portfolio: Portfolio) -> List[Stream[float]]:
        """Creates a list of streams to describe a :class:`Portfolio`.

        :param portfolio: The portfolio to create the streams for.
        :type portfolio: Portfolio
        :return: A list of streams to describe a :class:`Portfolio`.
        :rtype: List[Stream[float]]
        """
        base_symbol = portfolio.base_instrument.symbol
        sources = []

        for wallet in portfolio.wallets:
            symbol = wallet.instrument.symbol
            sources += wallet.exchange.streams()
            sources += FeedController.create_wallet_source(wallet, include_worth=(symbol != base_symbol))

        worth_streams = []
        for s in sources:
            if s.name.endswith(base_symbol + ":/total") or s.name.endswith("worth"):
                worth_streams += [s]

        net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
        sources += [net_worth]

        return sources