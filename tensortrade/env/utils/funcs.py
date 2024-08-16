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
from __future__ import annotations

import typing

from tensortrade.feed import Stream, NameSpace

if typing.TYPE_CHECKING:
    from typing import List

    from tensortrade.oms.wallets import Portfolio, Wallet

def create_wallet_source(wallet: Wallet, include_worth: bool = True) -> List[Stream[float]]:
    """Creates a list of streams to describe a `Wallet`.

    Parameters
    ----------
    wallet : `Wallet`
        The wallet to make streams for.
    include_worth : bool, default True
        Whether or

    Returns
    -------
    `List[Stream[float]]`
        A list of streams to describe the `wallet`.
    """
    exchange_name = wallet.exchange.name
    symbol = wallet.instrument.symbol

    streams = []

    with NameSpace(exchange_name + ":/" + symbol):
        free_balance = Stream.sensor(wallet, lambda w: w.balance.as_float(), dtype="float").rename("free")
        locked_balance = Stream.sensor(wallet, lambda w: w.locked_balance.as_float(), dtype="float").rename("locked")
        total_balance = Stream.sensor(wallet, lambda w: w.total_balance.as_float(), dtype="float").rename("total")

        streams += [free_balance, locked_balance, total_balance]

        if include_worth:
            price = Stream.select(wallet.exchange.streams(), lambda node: node.name.endswith(symbol))
            worth = price.mul(total_balance).rename('worth')
            streams += [worth]

    return streams

def create_internal_streams(portfolio: Portfolio) -> List[Stream[float]]:
    """Creates a list of streams to describe a `Portfolio`.

    Parameters
    ----------
    portfolio : `Portfolio`
        The portfolio to make the streams for.

    Returns
    -------
    `List[Stream[float]]`
        A list of streams to describe the `portfolio`.
    """
    base_symbol = portfolio.base_instrument.symbol
    sources = []

    for wallet in portfolio.wallets:
        symbol = wallet.instrument.symbol
        sources += wallet.exchange.streams()
        sources += create_wallet_source(wallet, include_worth=(symbol != base_symbol))

    worth_streams = []
    for s in sources:
        if s.name.endswith(base_symbol + ":/total") or s.name.endswith("worth"):
            worth_streams += [s]

    net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
    sources += [net_worth]

    return sources