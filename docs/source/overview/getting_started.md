## Getting Started

You can get started testing on Google Colab or your local machine, by viewing our [many examples](https://github.com/erhardtconsulting/tensortrade-ng/tree/main/examples)

---

## Installation

TensorTrade-NG recommends Python >= 3.12.0 for all functionality to work as expected.

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