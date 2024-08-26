# Code Structure

The TensorTrade library is modular. The `tensortrade-ng` library usually has a
common setup when using components. If you wish to make a particular class a
component all you need to do is subclass `Component`.

```python
from tensortrade.core.component import Component


class Example(Component):
    """An example component to show how to subclass."""

    def foo(self, arg1, arg2) -> str:
        """A method to return a string."""
        raise NotImplementedError()

    def bar(self, arg1, arg2, **kwargs) -> int:
        """A method to return an integer."""
```

From this abstract base class, more concrete and custom subclasses can be made
that provide the implementation of these methods.

<br>**Example of Structure**<br>
A good example of this structure is the `AbstractRewardScheme` component. This component
controls the reward mechanism of a `TradingEnv`.

The beginning of the code in [AbstractRewardScheme](https://github.com/erhardtconsulting/tensortrade-ng/blob/main/src/tensortrade/env/rewards/abstract.py) is seen here.

```python
from __future__ import annotations

from abc import abstractmethod

from tensortrade.core.base import TimeIndexed
from tensortrade.core.component import Component
from tensortrade.env.mixins.scheme import SchemeMixin


class AbstractRewardScheme(SchemeMixin, Component, TimeIndexed):
    """A component to compute the reward at each step of an episode."""

    registered_name = "rewards"

    @abstractmethod
    def reward(self) -> float:
        """Computes the reward for the current step of an episode.

        :return: The computed reward.
        :rtype: float
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the reward scheme."""
        pass
```

As you can see above, the [RewardScheme](https://github.com/erhardtconsulting/tensortrade-ng/blob/main/src/tensortrade/env/rewards/abstract.py) has a majority of the
structural and mechanical details that guide all other representations of that
type of class. When creating a new reward scheme, one needs to add further
details for how information from then environment gets converted into a reward.
