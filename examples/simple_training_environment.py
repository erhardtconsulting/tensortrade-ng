"""
This creates a simple training environment.
"""

import pandas as pd
import numpy as np
import ta.trend
import ta.momentum

from stable_baselines3 import PPO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensortrade.env.renderers import TerminalRenderer
from tensortrade.env.stoppers import MaxLossStopper
from tensortrade.feed import Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from tensortrade.env import TradingEnv
from tensortrade.env.observers import SimpleObserver
from tensortrade.env.actions import BSH
from tensortrade.env.rewards import SimpleProfit
from tensortrade.env.plotters import PlotlyTradingChart
from tensortrade.feed import DataFeed


"""
Loading data
"""

df = pd.read_csv('data/BTC_USDT_5m_20240601-20240731.csv').set_index('timestamp')

"""
Create TA features
"""

# Simple Moving Averages (SMA)
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()

# Exponential Moving Averages (EMA)
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

# Relative Strength Index (RSI)
df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

# Moving Average Convergence Divergence (MACD)
macd = ta.trend.MACD(df['close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()

# Price change
df['price_change'] = df['close'].pct_change()

"""
Copy OHLCV data
"""

df['raw-open'] = df['open']
df['raw-high'] = df['high']
df['raw-low'] = df['low']
df['raw-close'] = df['close']
df['raw-volume'] = df['volume']

"""
Clean up and prepare data for learning
"""

# Remove empty values
df.dropna(inplace=True)

# Split in raw data and feature data
feature_columns = [feature for feature in list(df) if not feature.startswith('raw-')]
df_features = df[feature_columns]
df_raw = df.drop(columns=feature_columns)

# Normalize feature data
scaler = StandardScaler()
df_features = scaler.fit_transform(df_features)
df_features = pd.DataFrame(df_features, columns=feature_columns, index=df.index)

# Concat dataframe again
df = pd.concat([df_features, df_raw], axis=1)

# Convert dataframe to float32
df = df.astype(np.float32)

# Last but not least we split the data frame into training and testing data
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=False)

"""
Create Portfolio
"""

# Prepare trading instruments
USDT = Instrument('USDT', 2, 'US Dollar Tether')
BTC = Instrument('BTC', 6, 'Bitcoin')

# prepare exchange
prices_stream = Stream.source(df['raw-close'], dtype='float').rename('USDT/BTC')
exchange = Exchange('dummy', service=execute_order)(prices_stream)

# prepare wallets
usdt_wallet = Wallet(exchange, 1000 * USDT)
btc_wallet = Wallet(exchange, 0 * BTC)

# prepare portfolio
portfolio = Portfolio(USDT, [
    usdt_wallet,
    btc_wallet
])

"""
Train Agent / Build TensorTrade-NG environment
"""

raw_data = ['raw-open', 'raw-high', 'raw-low', 'raw-close', 'raw-volume']

# prepare features
features = [Stream.source(train_df[f], dtype="float").rename(f) for f in feature_columns]

# prepare action scheme
action_scheme = BSH(cash=usdt_wallet, asset=btc_wallet) # Buy, Sell, Hold Action Scheme

# prepare meta feed
meta = [Stream.source(train_df.index).rename('date')]
meta += [Stream.source(train_df[f], dtype="float").rename(f[4:]) for f in raw_data]
meta += [Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")]

feed = DataFeed([
    Stream.group(features).rename('features'),
    Stream.group(meta).rename('meta')
])

# create the tensortrade environment
env = TradingEnv(
    portfolio=portfolio,
    feed=feed,
    action_scheme=action_scheme,
    reward_scheme=SimpleProfit(), # Reward on profit
    observer=SimpleObserver(), # Only show one observation at time
    stopper=MaxLossStopper(max_allowed_loss=0.5), # Stop when loosing more than 50%
    renderer=TerminalRenderer(), # Render to terminal
    render_mode='human' # Enable rendering
)

# Last but not least create our model and learn it
PPO('MlpPolicy', env, verbose=1).learn(10_000)