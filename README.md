# Language Processing Feature Analysis and Classification

This project focuses on analyzing and classifying language processing features using machine learning models. It involves calculating feature differences, training classifiers, and evaluating model performance on various validation sets. Key methods include logistic regression, random forest, and Bayesian optimization for hyperparameter tuning.

## Project Structure

- **Feature Calculation**: Computes absolute differences between various text processing features, such as POS density and dependency features, across text samples.
- **Model Training**: Implements logistic regression and random forest classifiers to detect author changes based on linguistic feature differences.
- **Bayesian Optimization**: Uses Bayesian optimization to optimize the random forest model’s hyperparameters for improved classification performance.
  
## Prerequisites

- Python 3.6+
- Required packages: `pandas`, `numpy`, `scikit-learn`, `bayes_opt` (for Bayesian Optimization)

Install packages using:
```bash
pip install pandas numpy scikit-learn bayesian-optimization
