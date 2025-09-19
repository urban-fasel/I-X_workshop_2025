# I-X_workshop_2025

# 📉 SINDy Introductory Tutorial

Welcome to the **SINDy (Sparse Identification of Nonlinear Dynamics)** introductory tutorial!

## Run in your browser
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-username/sindy-intro-tutorial/HEAD?labpath=1_data_generation.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/sindy-intro-tutorial/blob/HEAD/1_data_generation.ipynb)



## What is SINDy?

**SINDy** is a machine learning algorithm used to discover the underlying **governing equations** of a dynamic system directly from data. Unlike traditional methods that require prior knowledge of the system's physics, SINDy works by building a library of candidate functions and then using a sparse regression algorithm to find the simplest combination of those functions that describes the data. This allows for the automatic discovery of **parsimonious** (simple yet powerful) models.

### Why is it useful?

* **Model Discovery:** Find the differential equations that govern a system's behavior without needing to guess them beforehand.
* **Interpretability:** The resulting models are symbolic and easy to interpret, unlike black-box models like neural networks.
* **Extensibility:** Can be applied to a wide range of systems, from fluid dynamics to neuroscience.

## Tutorial Structure

This repository is structured to walk you through a complete example, from generating data to identifying the equations.

1.  **`1_data_generation.ipynb`**: Generate data for a simple dynamic system (e.g., the Lotka-Volterra predator-prey model).
2.  **`2_sindy_implementation.ipynb`**: Implement the SINDy algorithm from scratch using `numpy` and `scipy`.
3.  **`3_sindy_with_pySINDy.ipynb`**: Show how to use the dedicated `pySINDy` library, which simplifies the process.
4.  **`4_sensitivity_analysis.ipynb`**: An interactive tutorial exploring the sensitivity of SINDy to **noise**, **data length**, and **sampling rate**.
5.  **`utils.py`**: Helper functions for plotting.
6.  **`requirements.txt`**: A list of the necessary Python libraries.

## Getting Started

1.  Clone this repository: `git clone https://github.com/your-username/sindy-intro-tutorial.git`
2.  Navigate to the directory: `cd sindy-intro-tutorial`
3.  Install the required packages: `pip install -r requirements.txt`
4.  Open and run the Jupyter notebooks in order.

Enjoy your journey into the world of SINDy! 🚀
