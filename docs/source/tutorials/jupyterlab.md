# Using TensorTrade-NG with JupyterLab

Explore the power of TensorTrade-NG, a flexible framework designed to facilitate the development of reinforcement 
learning algorithms in financial trading environments. This tutorial will guide you through setting up your environment 
in JupyterLab, ensuring you have everything you need to start experimenting with custom trading strategies.

## Prerequisites
- Python 3.12
- [JupyterLab](https://jupyter.org/)

## Preparation

1. **Create a new Python virtual environment and activate it**
   ```sh
   cd /my/project/directory
   python -m venv venv
   source venv/bin/activate
   ```
   *This step isolates your TensorTrade-NG environment, ensuring no conflicts with other Python packages.*
2. **Install TensorTrade-NG with JupyterLab extra packages and create a new kernel for Jupyter**
   ```sh
   pip install "tensortrade-ng[jlab]"
   ipython kernel install --user --name tensortrade-ng
   ```
   *By including the **jlab** extras, you ensure compatibility with JupyterLab and streamline your development process by 
   having the necessary tools and dependencies readily available.*
3. **Restart JupyterLab and use the new kernel** \
   *After restarting JupyterLab, you should see the new kernel "tensortrade-ng" available. Select it to begin your 
   TensorTrade-NG projects. This kernel is pre-configured with all the necessary dependencies to make your first steps 
   in TensorTrade-NG effortless.*

## What You Learned

In this tutorial, you learned how to set up a dedicated Python environment for TensorTrade-NG, ensuring that your 
development environment is clean, organized, and tailored for machine learning projects. By installing TensorTrade-NG 
with JupyterLab extras, you ensured that your environment is equipped to handle the interactive development of financial 
trading strategies using reinforcement learning.
