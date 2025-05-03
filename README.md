# pyspark-credit-risk-scoring
PySpark-based credit scoring project simulating DE AM-level workflows and DQ checks. Implements end-to-end data pipeline and classification models (RF, GBT, LR) to assess credit risk exposure. Emphasizes scalable feature engineering and model evaluation.

## Overview

This repository contains a PySpark-based machine learning pipeline designed to predict credit scores using a dataset with features related to financial and demographic information. The pipeline includes data loading, preprocessing, feature engineering, model training, and evaluation, with a focus on data quality (DQ) checks to ensure the reliability of the dataset before training.

## Table of Contents

* [Project Setup](#project-setup)
* [Code Walkthrough](#code-walkthrough)

  * [1. Data Loading](#1-data-loading)
  * [2. Data Preprocessing](#2-data-preprocessing)
  * [3. Data Quality (DQ) Checks](#3-data-quality-dq-checks)
  * [4. Feature Engineering](#4-feature-engineering)
  * [5. Model Training](#5-model-training)
  * [6. Model Evaluation](#6-model-evaluation)
* [Dependencies](#dependencies)
* [Usage](#usage)

## Project Setup

Before you start, ensure that you have the necessary dependencies installed. You can install them using the following:

```bash
pip install -r requirements.txt
```

## Code Walkthrough

### 1. Data Loading

We begin by loading the raw credit scoring data into a PySpark DataFrame. This step involves reading the data from a CSV file or any other compatible data source.

```python
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("CreditScoringPipeline").getOrCreate()

# Load the dataset
data_path = "path_to_data.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.show(5)
```

**Explanation**:
We initialize a Spark session and load the data from a CSV file, inferring the schema for automatic data type detection. The `show()` method is used to preview the first few rows of the dataset.

### 2. Data Preprocessing

In this step, we clean the data by handling missing values and performing necessary transformations.

```python
from pyspark.sql.functions import col

# Handle missing values by filling with a default value or dropping rows
df_cleaned = df.fillna({"column_name": "default_value"})

# Convert categorical columns to numeric using StringIndexer or OneHotEncoder
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="categorical_column", outputCol="indexed_column")
df_cleaned = indexer.fit(df_cleaned).transform(df_cleaned)

df_cleaned.show(5)
```

**Explanation**:

* Missing values are handled using `fillna()`, where we specify default values for columns with missing data.
* We use the `StringIndexer` to convert categorical variables into numeric indices, which is necessary for machine learning algorithms.

### 3. Data Quality (DQ) Checks

Before proceeding with the analysis, it’s essential to ensure the quality of the data by checking for inconsistencies or outliers.

```python
# Check for duplicate rows
df_cleaned = df_cleaned.dropDuplicates()

# Check for missing values after imputation
missing_values = df_cleaned.select([col(c).isNull().alias(c) for c in df_cleaned.columns]).agg({"*": "sum"})
missing_values.show()

# Data quality checks: Range checks for numerical columns
df_cleaned.filter((col("age") < 18) | (col("age") > 100)).show()
```

**Explanation**:

* Duplicates are removed using `dropDuplicates()`.
* We verify if there are any remaining missing values after handling them by inspecting the sum of nulls in each column.
* We perform range checks for numerical columns (e.g., age should be between 18 and 100).

### 4. Feature Engineering

Feature engineering is crucial to improve model performance. Here, we create new features or transform existing ones based on domain knowledge.

```python
from pyspark.ml.feature import VectorAssembler

# Assemble feature vector for model input
feature_columns = ["age", "income", "credit_history"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_features = assembler.transform(df_cleaned)

df_features.select("features").show(5)
```

**Explanation**:
The `VectorAssembler` is used to combine multiple feature columns (such as `age`, `income`, `credit_history`) into a single feature vector, which is required by most PySpark machine learning algorithms.

### 5. Model Training

We now split the data into training and testing datasets and train a machine learning model, such as Logistic Regression.

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Split the data into training and testing sets
train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=1234)

# Initialize the model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Set up hyperparameter tuning with cross-validation
param_grid = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.1, 0.01])
              .addGrid(lr.maxIter, [10, 20])
              .build())

evaluator = BinaryClassificationEvaluator()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3)

# Train the model using cross-validation
model = crossval.fit(train_data)
```

**Explanation**:

* The dataset is split into training and testing sets (80%/20%).
* A Logistic Regression model is initialized, and hyperparameter tuning is performed using `CrossValidator` to select the best parameters (`regParam` and `maxIter`).
* The model is trained using the training data.

### 6. Model Evaluation

Once the model is trained, we evaluate its performance on the test data.

```python
# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using ROC-AUC score
roc_auc = evaluator.evaluate(predictions)
print(f"ROC-AUC: {roc_auc}")
```

**Explanation**:
We use the `BinaryClassificationEvaluator` to compute the ROC-AUC score, which is a common metric for evaluating classification models.

## Dependencies

* PySpark
* pandas
* numpy
* scikit-learn

You can install these dependencies via the following command:

```bash
pip install pyspark pandas numpy scikit-learn
```

## Usage

To run the pipeline, simply execute the script or load the notebook with the dataset you want to analyze. Ensure that your environment is properly set up with PySpark and the necessary dependencies.

