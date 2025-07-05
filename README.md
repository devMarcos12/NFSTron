# NFStron - Neural Networks from Scratch

A from-scratch implementation of neural networks in Python, built for educational purposes to demonstrate the foundational principles of machine learning and deep learning.

## About the Project

NFStron is a minimalist library that implements Perceptrons and Multi-Layer Perceptrons (MLPs) without relying on external frameworks like TensorFlow or PyTorch. This project was developed with an educational focus, aiming to provide a deep understanding of the fundamental algorithms that power neural networks.

## Key Features

- **Simple Perceptron**: The basic artificial neuron trained using the Perceptron learning rule.
- **Multi-Layer Perceptron (MLP)**: A multi-layered neural network with a from-scratch implementation of the backpropagation algorithm.
- **Activation Functions**: Includes Step, Sigmoid, Tanh, and ReLU activation functions.
- **Classic Problems**: Comes with experiments for classic logical gates: AND, OR, and XOR.

## Getting Started

To get the project up and running on your local machine, follow these steps.

### Prerequisites

- Python 3.13 or higher.
- `uv` package installer. If you don't have `uv`, you can install it with:
  ```bash
  pip install uv
  ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/NFStron.git
    cd NFStron
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management, which is a very fast and modern Python package manager. To install the dependencies, run:
    ```bash
    uv sync
    ```
    This command will install the exact versions of the dependencies specified in the `uv.lock` file, which were resolved from `pyproject.toml`. This ensures a consistent and reproducible development environment.

## Project Structure

The project is organized into the following directories:

-   `perceptron/`: Contains the core logic for the neural network, including the implementation of neurons, activation functions, and the MLP.
-   `dataset/`: Includes modules for generating or loading datasets, such as the logical problems for the experiments.
-   `experiments/`: Contains scripts to test and demonstrate the implemented neural network models (e.g., `test_and.py`, `test_or.py`, `test_xor.py`).
-   `visualizations/`: Includes scripts for plotting and visualizing the training process and results.

## How It Works

The core of the project is the `perceptron` module, which contains the building blocks of the neural network.

-   **`neuron.py`**: Defines the `Perceptron` class, which is the simplest form of a neural network.
-   **`mlp.py`**: Implements the `MLP` (Multi-Layer Perceptron), a feedforward neural network with one or more hidden layers. The training is done using the backpropagation algorithm, which is implemented from scratch.
-   **`activations.py`**: Provides various activation functions that can be used in the neurons.
-   **`utils.py`**: Contains utility functions used across the project.

## Usage

To run the experiments and see the neural network in action, you can execute the scripts in the `experiments` directory. For example, to run the XOR experiment:

```bash
python experiments/test_xor.py
```

This will train an MLP to solve the XOR problem and will likely generate a plot showing the training progress.
