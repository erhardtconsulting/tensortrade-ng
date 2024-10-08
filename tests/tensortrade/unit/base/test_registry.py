import warnings

import tensortrade.env.actions as actions
import tensortrade.env.rewards as rewards

warnings.filterwarnings("ignore")


def test_simple_actions():
    assert isinstance(actions.get('simple'), actions.SimpleOrders)


def test_managed_risk_actions():
    assert isinstance(actions.get('managed-risk'), actions.ManagedRiskOrders)


def test_simple_reward_scheme():
    assert isinstance(rewards.get('simple'), rewards.SimpleProfit)


def test_risk_adjusted_reward_scheme():
    assert isinstance(rewards.get('risk-adjusted'), rewards.RiskAdjustedReturns)
