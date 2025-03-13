# CSC 311 Fall 2023 Project

This repository contains the code, experiments, and documentation for our CSC 311 Fall 2023 project. Developed by Wo Ming Boaz Cheung, Ivan Ye, and Zexi Liu, the project explores various predictive modeling techniques—including k-Nearest Neighbor (kNN), Item Response Theory (IRT), Neural Networks, and Ensemble methods—to analyze and predict student responses. In addition, an extended deep neural network model with a deeper architecture, Leaky ReLU activation, and ADAM optimizer is implemented to capture more complex relationships in the data.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methods and Models](#methods-and-models)
  - [k-Nearest Neighbor](#k-nearest-neighbor)
  - [Item Response Theory (IRT)](#item-response-theory-irt)
  - [Neural Networks](#neural-networks)
  - [Ensemble Method](#ensemble-method)
  - [Extended Deep Neural Network](#extended-deep-neural-network)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [References](#references)
- [Contributors](#contributors)
- [License](#license)

## Overview

The goal of this project is to develop and compare multiple machine learning models for predicting student performance based on response data. We investigate:
- **Traditional methods:** kNN and IRT.
- **Modern approaches:** Neural networks and ensemble methods.
- **Advanced modeling:** An extended deep neural network with improved training via the ADAM optimizer and Leaky ReLU activation, utilizing GPU acceleration with CUDA.

## Project Structure

```
CSC311-Project/
├── data/                # Raw and processed data files
├── notebooks/           # Jupyter notebooks for experiments and visualizations
├── src/                 # Source code for model training and evaluation
├── results/             # Output results, figures, and logs from experiments
├── final_report.pdf     # Detailed final project report
└── README.md            # This file
```

## Methods and Models

### k-Nearest Neighbor
- **Objective:** Impute missing student responses using both user-based and item-based approaches.
- **Highlights:**  
  - Optimal user-based kNN (k = 11) achieved a test accuracy of approximately 68.42%.  
  - Optimal item-based kNN (k = 21) achieved a test accuracy of around 68.16%.

### Item Response Theory (IRT)
- **Objective:** Model student ability and question difficulty through a logistic framework.
- **Highlights:**  
  - Derived and optimized the log-likelihood function.  
  - Achieved a final test accuracy of about 70.84%.

### Neural Networks
- **Objective:** Leverage deep learning to model complex patterns in student response data.
- **Highlights:**  
  - Tuned hyperparameters such as epochs, learning rate, and regularization lambda.
  - Reached a final test accuracy of approximately 68.25%.

### Ensemble Method
- **Objective:** Combine predictions from multiple IRT models (trained on bootstrapped samples) to enhance robustness.
- **Highlights:**  
  - Improved test accuracy to about 71.10% through ensemble averaging.

### Extended Deep Neural Network
- **Objective:** Enhance model performance by deepening the architecture and optimizing with ADAM.
- **Highlights:**  
  - Integrated Leaky ReLU activations and CUDA-based parallel training.
  - Modified model achieved a test accuracy of roughly 70.25% with improved training dynamics.

## Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/CSC311-Project.git
   cd CSC311-Project
   ```

2. **Install Dependencies**
   - Ensure Python 3.8+ is installed.
   - Install required packages via:
     ```bash
     pip install -r requirements.txt
     ```
   - Dependencies include libraries for numerical computing (e.g., NumPy, SciPy), deep learning frameworks (e.g., TensorFlow or PyTorch), and data processing tools.

3. **Data Preparation**
   - Place the provided dataset files in the `data/` directory.
   - Run preprocessing scripts located in `src/` if necessary to format the data for training.

## Usage

- **Running Experiments:**  
  Open the Jupyter notebooks in the `notebooks/` directory to replicate experiments and view analyses.
- **Training Models:**  
  Execute scripts in the `src/` directory. For example:
  ```bash
  python src/train_knn.py
  python src/train_irt.py
  python src/train_nn.py
  python src/train_ensemble.py
  python src/train_extended_nn.py
  ```

## Results

Key experimental outcomes include:
- **kNN:** Test accuracies of ~68.42% (user-based) and ~68.16% (item-based).
- **IRT:** Test accuracy around 70.84%.
- **Neural Networks:** Test accuracy of ~68.25%.
- **Ensemble:** Improved test accuracy to ~71.10%.
- **Extended Deep Neural Network:** Test accuracy reached ~70.25%, with notable improvements in training efficiency via GPU acceleration.

For detailed graphs, training logs, and analysis, refer to the `results/` directory and the [final_report.pdf](final_report.pdf).

## Limitations

- **Data Sparsity:** The dataset's limited sample size and inherent sparsity may affect model robustness.
- **Hyperparameter Sensitivity:** Performance is highly sensitive to hyperparameter tuning.
- **Computational Demands:** Deep models, particularly those leveraging GPU acceleration, require substantial computational resources.

## References

- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
- [Other relevant references can be added here.]

## Contributors

- **Wo Ming Boaz Cheung:** Contributed to Part B (Formal Description, Figures/Diagrams, Comparison/Demonstration).
- **Zexi Liu:** Contributed to Part A (Neural Networks) and Part B (Limitations).
- **Ivan Ye:** Contributed to Part A (kNN, IRT, Ensemble).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
