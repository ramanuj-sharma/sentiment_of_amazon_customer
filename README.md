# Text Classification Project

This project focuses on building a text classification model to predict sentiment or helpfulness based on review data.

## Project Overview

- **Objective**: Develop a model to classify reviews as positive or negative based on their textual content and useful votes.
- **Techniques Used**:
  - TF-IDF Vectorization
  - Logistic Regression
  - Hyperparameter Tuning with GridSearchCV
  - Class Imbalance Handling with Random OverSampling

## Data Processing

1. **Text Vectorization**:
   - Used `TfidfVectorizer` to convert textual data into numerical features.
   - Removed English stop words to enhance feature relevance.

2. **Data Splitting**:
   - Split data into training and testing sets to evaluate model performance.

3. **Class Balancing**:
   - Applied `RandomOverSampler` to balance class distribution and improve model effectiveness.

## Model Building

- Utilized `LogisticRegression` with hyperparameter tuning via `GridSearchCV`.
- Adjusted model hyperparameters for optimal performance.
- Solver changed to `'liblinear'` to support both `l1` and `l2` regularization.

## Model Evaluation

- Assessed using confusion matrix and accuracy score.
  
  | Metric             | Value  |
  |--------------------|--------|
  | **Accuracy**       | 99.97% |
  | **True Positives** | 45,541 |
  | **True Negatives** | 774    |
  | **False Positives**| 0      |
  | **False Negatives**| 14     |

## Key Features

- **High Accuracy**: Demonstrated robust classification abilities.
- **Interpretability**: Analyzed top influential words contributing to model predictions.
- **Efficiency**: Applied RandomOverSampling to address class imbalance effectively.

## Getting Started

1. **Prerequisites**: Install necessary Python packages.

   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn