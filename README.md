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
```

## File Descriptions

- **`training_features_absolute.csv`**: Contains preprocessed training features.
- **`combined_train_labels.csv`**: Labels for training, indicating author change.
- **`validation_features.csv` and `validation_labels.csv`**: Feature and label files for easy, medium, and hard validation sets.

## Feature Processing

The script processes text features by:
1. Loading feature and label data.
2. Renaming columns and evaluating list-like columns.
3. Calculating absolute differences between feature values.

Run feature processing with:
```python
process_file(input_file, output_file)
```

## Model Training

### Logistic Regression
Trains a logistic regression model on the extracted features to classify author changes:
```python
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
```

### Random Forest with Bayesian Optimization
Uses Bayesian optimization to find the optimal hyperparameters for a random forest model:
```python
optimizer.maximize(init_points=10, n_iter=20)
```

Evaluates the model on each validation set using metrics such as F1 score, accuracy, precision, and recall.

## Usage

1. **Feature Calculation**:  
   Update paths in `files_to_process` and run `process_file(input_file, output_file)` for each file.

2. **Model Training and Evaluation**:  
   Modify paths in `train_features_path` and `val_features_path` for validation datasets, then run the training and evaluation cells.

3. **Hyperparameter Optimization**:  
   Run Bayesian optimization on the random forest model by adjusting `init_points` and `n_iter`.

## Output

The model generates:
- Classification performance metrics for each validation set.
- Feature importance rankings for random forest models.

## License
This project is licensed under the MIT License.
```
