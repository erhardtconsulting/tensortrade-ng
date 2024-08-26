[![Apache License](https://img.shields.io/github/license/erhardtconsulting/tensortrade-ng.svg?color=brightgreen)](http://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

---

<div align="center">
  <img src="https://github.com/notadamking/tensortrade/blob/master/docs/source/_static/logo.jpg">
</div>

---

> **ℹ️ TensorTrade-NG was forked from the [TensorTrade](https://github.com/tensortrade-org/tensortrade)-Project, mainly because the code needed a lot refactoring, was outdated and it looked not really maintained anymore. Therefor we did a lot of breaking changes, removed old unused stuff and cleaned up. We tried to preserve the APIs but if you want to switch from TensorTrade to TensorTrade-NG be aware that it may take a little bit of effort. Apart from that we thank all the former developers and community for their awesome work and are happy to welcome them here.**

**TensorTrade-NG is still in Beta, meaning it should be used very cautiously if used in production, as it may contain bugs.**

TensorTrade-NG is an open source Python framework for building, training, evaluating, and deploying robust trading algorithms using reinforcement learning. The framework focuses on being highly composable and extensible, to allow the system to scale from simple trading strategies on a single CPU, to complex investment strategies run on a distribution of HPC machines.

Under the hood, the framework uses many of the APIs from existing machine learning libraries to maintain high quality data pipelines and learning models. One of the main goals of TensorTrade is to enable fast experimentation with algorithmic trading strategies, by leveraging the existing tools and pipelines provided by `numpy`, `pandas` and `gymnasium`. The idea behind Tensorflow-NG is not to implement all the machine learning stuff itself. But to provide a solid framework that makes it possible to quickly provide a working environment for other tools such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io).

Every piece of the framework is split up into re-usable components, allowing you to take advantage of the general use components built by the community, while keeping your proprietary features private. The aim is to simplify the process of testing and deploying robust trading agents using deep reinforcement learning, to allow you and I to focus on creating profitable strategies.

_The goal of this framework is to enable fast experimentation, while maintaining production-quality data pipelines._

Read [the documentation](https://tensortrade-ng.io/).

## Guiding principles

_Inspired by [Keras' guiding principles](https://github.com/keras-team/keras)._

- **User friendliness.** TensorTrade is an API designed for human beings, not machines. It puts user experience front and center. TensorTrade follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

- **Modularity.** A trading environment is a conglomeration of fully configurable modules that can be plugged together with as few restrictions as possible. In particular, exchanges, feature pipelines, action schemes, reward schemes, trading agents, and performance reports are all standalone modules that you can combine to create new trading environments.

- **Easy extensibility.** New modules are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making TensorTrade suitable for advanced research and production use.

## Getting Started

You can get started testing on Google Colab or your local machine, by viewing our [many examples](https://github.com/erhardtconsulting/tensortrade-ng/tree/master/examples).

**Recommended beginning points:**

* [Sample Environment as Python Script](https://github.com/erhardtconsulting/tensortrade-ng/blob/main/examples/simple_training_environment.py)
* 

## Installation

TensorTrade-NG requires Python >= 3.12.0 for all functionality to work as expected.

### As package
You can install TensorTrade-NG both as a pre-packaged solution by running the default setup command.
```bash
pip install tensortrade-ng
```

### Via git
You can also alternatively install TensorTrade-NG directly from the master code repository, pulling directly from the latest commits. This will give you the latest features/fixes, but it is highly untested code, so proceed at your own risk.
```bash
pip install git+https://github.com/erhardtconsulting/tensortrade-ng.git
```

### Cloning the repository

> **⚠️ Warning**: This repository uses *git-lfs* for storing the Jupyter Notebooks and other big files. Make sure to install the [git-lfs Extension](https://git-lfs.com/) before cloning the repository.

You can clone/download the repository in your local environment and manually install the requirements, either the "base" ones, or the ones that also include requirements to run the examples in the documentation.

```bash
# install only base requirements
pip install -e .

# install all requirements
pip install -e ".[dev]"
```

### Build Documentation

You can either build the documentation once or serve it locally.

> **Prerequisites:** You need to have [pandoc](https://pandoc.org/installing.html) installed locally for converting jupyter notebooks. Otherwise it won't work. The *pip*-version won't work, because it's just a wrapper. You need to use your package manager, like `brew` or `apt`. 

**Run documentation as local webserver**

```bash
hatch run docs:serve
```

**Build documentation**

```bash
hatch run docs:build
```

### Run Test Suite

To run the test suite, execute the following command.

```bash
# Test all
hatch run test:run

# Test only specific python version
hatch run +py=3.12 test:run

# Test with coverage (only one python version recommended)
hatch run +py=3.12 test:run-cov
```

## Support

You can also post **bug reports and feature requests** in [GitHub issues](https://github.com/erhardtconsulting/tensortrade-ng/issues). Make sure to read [our guidelines](https://github.com/erhardtconsulting/tensortrade-ng/blob/master/CONTRIBUTING.md) first.

If you have **questions or anything else** that needs to be discussed. Please use [GitHub Discussions](https://github.com/erhardtconsulting/tensortrade-ng/discussions) rather than opening an issue.


## Contributors

Contributions are encouraged and welcomed. This project is meant to grow as the community around it grows. Let us know on [GitHub Discussions](https://github.com/erhardtconsulting/tensortrade-ng/discussions) if there is anything that you would like to see in the future, or if there is anything you feel is missing.

**Working on your first Pull Request?** You can learn how from this _free_ series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).

![https://github.com/erhardtconsulting/tensortrade-ng/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=erhardtconsulting/tensortrade-ng)
