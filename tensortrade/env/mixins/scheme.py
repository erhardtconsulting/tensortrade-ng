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

if typing.TYPE_CHECKING:
    from typing import Optional

    from tensortrade.env import TradingEnv

class SchemeMixin:
    """A mixin class that provides managed access to a :class:`TradingEnv` instance.

    This mixin ensures that any class inheriting from it has a `trading_env` attribute
    that is properly initialized before use. The `trading_env` attribute is managed
    through a property to enforce this initialization check.

    :param _trading_env: A :class:`TradingEnv` instance, defaults to None
    :type _trading_env: :class:`TradingEnv`, optional
    """
    def __init__(self):
        self._trading_env: Optional[TradingEnv] = None

    @property
    def trading_env(self) -> TradingEnv:
        """Provides access to the :class:`TradingEnv` instance.

        This property checks whether `_trading_env` has been initialized.
        If `_trading_env` is `None`, a `ValueError` is raised to prevent the use of an uninitialized :class:`TradingEnv`.

        :return: The initialized :class:`TradingEnv` instance.
        :rtype: :class:`TradingEnv`
        :raises ValueError: If `_trading_env` is not initialized.
        """
        if self._trading_env is None:
            raise ValueError("Trading environment is not initialized. Set trading_env first.")
        return self._trading_env

    @trading_env.setter
    def trading_env(self, value: TradingEnv):
        """Sets the :class:`TradingEnv` instance.

        This setter allows for the initialization of the `_trading_env` attribute.

        :param value: The `TradingEnv` instance to be assigned to `_trading_env`.
        :type value: :class:`TradingEnv`
        """
        self._trading_env = value