:# Linear Regression with Lasso and Rigde

## Project Description

This project implements a custom linear regression model using TensorFlow and Keras. It also includes two custom loss functions based on mean square error, one for Lasso regression and another for Ridge regression.

### Features

- Custom linear regression model.
- Custom loss functions for Lasso and Ridge regression.
- Utilizes TensorFlow and Keras.

## Getting Started

Follow these steps to get started with the project:

### Prerequisites

Make sure you have the necessary packages installed by running:

pip install -r requirements.txt

## Usage

You can use the custom linear regression model in two ways:

**Standalone Usage**:

1. Use the `LinearRegression.py` file for standalone usage.
2. Ensure you preprocess your data appropriately before feeding it into the model.

**Integration into Main Program**:

1. Make necessary adjustments to the `main_program.py` file to integrate the model into your workflow.
2. Note that the data pipeline in the main program assumes a CSV file that is split into training and validation datasets.
3. You should perform preprocessing on your data before using it in this program.
