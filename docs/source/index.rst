|Logo|

.. admonition:: Fork of TensorTrade
   :class: seealso

   TensorTrade-NG has been forked from the `TensorTrade <https://github.com/tensortrade-org/tensortrade/>`_-Project,
   mainly because the code needed a lot refactoring, was outdated and it looked not really maintained anymore.
   Therefor we did a lot of breaking changes, removed old unused stuff and cleaned up. We tried to preserve
   the APIs but if you want to switch from TensorTrade to TensorTrade-NG be aware that it may take a little
   bit of effort. Apart from that we thank all the former developers and community for their awesome work and
   are happy to welcome them here.

`TensorTrade-NG`_ is an open source Python framework for building,
training, evaluating, and deploying robust trading algorithms using
reinforcement learning. The framework focuses on being highly composable
and extensible, to allow the system to scale from simple trading
strategies on a single CPU, to complex investment strategies run on a
distribution of HPC machines.

Under the hood, the framework uses many of the APIs from existing machine
learning libraries to maintain high quality data pipelines and learning models.
One of the main goals of TensorTrade is to enable fast experimentation
with algorithmic trading strategies, by leveraging the existing tools
and pipelines provided by `numpy`, `pandas` and `gymnasium`. The idea
behind Tensorflow-NG is not to implement all the machine learning stuff
itself. But to provide a solid framework that makes it possible to quickly
provide a working environment for other tools such as `Stable-Baselines3`_.

Every piece of the framework is split up into re-usable components,
allowing you to take advantage of the general use components built by
the community, while keeping your proprietary features private. The aim
is to simplify the process of testing and deploying robust trading
agents using deep reinforcement learning, to allow you and I to focus on
creating profitable strategies.

*The goal of this framework is to enable fast experimentation, while
maintaining production-quality data pipelines.*

Feel free to also walk through the `Medium tutorial`_.

The most up to date example is `Train and Evaluate using Ray`_. We suggest you to start there!

Guiding principles
------------------

*Inspired by* `Keras' guiding principles`_.

*User friendliness.* TensorTrade is an API designed for human beings,
not machines. It puts user experience front and center. TensorTrade
follows best practices for reducing cognitive load: it offers consistent
& simple APIs, it minimizes the number of user actions required for
common use cases, and it provides clear and actionable feedback upon
user error.

*Modularity.* A trading environment is a conglomeration of fully
configurable modules that can be plugged together with as few
restrictions as possible. In particular, exchanges, feature pipelines,
action schemes, reward schemes, trading agents, and performance reports
are all standalone modules that you can combine to create new trading
environments.

*Easy extensibility.* New modules are simple to add (as new classes and
functions), and existing modules provide ample examples. To be able to
easily create new modules allows for total expressiveness, making
TensorTrade suitable for advanced research and production use.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io
.. _TensorTrade-NG: https://github.com/erhardtconsulting/tensortrade-ng
.. _Medium tutorial: https://medium.com/@notadamking/trade-smarter-w-reinforcement-learning-a5e91163f315
.. _Keras' guiding principles: https://github.com/keras-team/keras
.. _Train and Evaluate using Ray: examples/train_and_evaluate_using_ray.html

.. |Logo| image:: _static/logo.svg


.. toctree::
    :maxdepth: 1
    :hidden:

    Home <self>


.. toctree::
    :maxdepth: 1
    :caption: Overview

    overview/getting_started.md


.. toctree::
    :maxdepth: 1
    :caption: Examples

    examples/overview.md
    examples/setup_environment_tutorial.md
    examples/train_and_evaluate_using_ray.md
    examples/renderers_and_plotly_chart.md
    examples/ledger_example.md


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials/jupyterlab.md
    tutorials/setup_base_environment.ipynb


.. toctree::
    :maxdepth: 1
    :caption: Components

    components/components.md
    components/action_scheme.md
    components/reward_scheme.md
    components/observer.md
    components/stopper.md
    components/informer.md
    components/renderer.md


.. toctree::
    :maxdepth: 1
    :caption: Environments

    envs/overview.md


.. toctree::
    :maxdepth: 1
    :caption: Feed

    feed/overview.md

.. toctree::
    :maxdepth: 1
    :caption: Order Management System

    oms/overview.md


.. toctree::
    :maxdepth: 1
    :caption: Agents

    agents/overview.md
    agents/well_performing_agent.md


.. toctree::
    :maxdepth: 1
    :caption: API reference

    API reference <api/modules>
