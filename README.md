# PySpark Credit Risk Scoring Model

This repository contains a PySpark-based pipeline for credit risk scoring. The model uses a variety of machine learning classifiers, including Random Forest, Gradient Boosted Trees, and Logistic Regression, to predict whether a person is at risk of serious delinquency in the next two years based on financial features. The project leverages PySpark for distributed data processing and machine learning.

## Project Overview

* **Objective**: Predict the likelihood of a person experiencing serious delinquency within the next 2 years based on historical financial data.
* **Technology**: PySpark, Machine Learning, SQL, Data Engineering
* **Dataset**: The model is built using the Kaggle "Give Me Some Credit" dataset.

## Prerequisites

Before running the code, ensure that you have the following installed:

* **Apache Spark**: Set up a Spark cluster (local or distributed).
* **PySpark**: The Python API for Spark.

```bash
pip install pyspark
```

## Step-by-Step Implementation

### 1. Initialize Spark Context and HiveContext

```python
from pyspark.sql import HiveContext
from pyspark import SparkContext

sc = SparkContext()
hc = HiveContext(sc)
print(hc)
```

### 2. Data Loading

We begin by downloading and loading the data into a distributed data frame using a custom CSV parser. The data file is fetched from a remote URL:

```python
import os

DATA_FILE_NAME = 'CreditScoring.csv'
DATA_URL = 'https://raw.githubusercontent.com/ChicagoBoothML/DATA___Kaggle___GiveMeSomeCredit/master/%s' % DATA_FILE_NAME
os.system('curl %s --output %s' % (DATA_URL, DATA_FILE_NAME))

# Load CSV data into a distributed data frame
from pyspark_csv import csvToDataFrame
credit_scoring_ddf = csvToDataFrame(
    sqlCtx=hc,
    rdd=sc.textFile(DATA_FILE_NAME),
    columns=None,
    sep=',',
    parseDate=True
).cache()
credit_scoring_ddf.registerTempTable('credit_scoring')
credit_scoring_ddf.printSchema()
```

### 3. Data Preprocessing

We process the data by removing redundant columns, converting labels to strings, and ensuring all numerical columns are of type `DoubleType`.

```python
credit_scoring_ddf = hc.sql(
    """
    SELECT 
        CASE WHEN SeriousDlqin2yrs > 0 THEN 'yes' ELSE 'no' END AS SeriousDlqin2yrs,
        CAST(RevolvingUtilizationOfUnsecuredLines AS DOUBLE) AS RevolvingUtilizationOfUnsecuredLines,
        CAST(age AS DOUBLE) AS age,
        CAST(NumberOfTime30-59DaysPastDueNotWorse AS DOUBLE) AS NumberOfTime30-59DaysPastDueNotWorse,
        CAST(DebtRatio AS DOUBLE) AS DebtRatio,
        CAST(MonthlyIncome AS DOUBLE) AS MonthlyIncome,
        CAST(NumberOfOpenCreditLinesAndLoans AS DOUBLE) AS NumberOfOpenCreditLinesAndLoans,
        CAST(NumberOfTimes90DaysLate AS DOUBLE) AS NumberOfTimes90DaysLate,
        CAST(NumberRealEstateLoansOrLines AS DOUBLE) AS NumberRealEstateLoansOrLines,
        CAST(NumberOfTime60-89DaysPastDueNotWorse AS DOUBLE) AS NumberOfTime60-89DaysPastDueNotWorse,
        CAST(NumberOfDependents AS DOUBLE) AS NumberOfDependents
    FROM credit_scoring
    """
).cache()
credit_scoring_ddf.registerTempTable('credit_scoring')
credit_scoring_ddf.printSchema()
```

### 4. Data Splitting

We split the data into training and testing sets:

```python
credit_scoring_train_ddf, credit_scoring_test_ddf = credit_scoring_ddf.randomSplit(
    weights=[.3, .7], seed=99
)

credit_scoring_train_ddf.cache()
credit_scoring_test_ddf.cache()
```

### 5. Model Pipeline Construction

We create a pipeline for each classifier:

#### Random Forest Classifier

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer

# Feature assembler
feature_vector_assembler = VectorAssembler(inputCols=X_var_names, outputCol='features')

# Random Forest classifier
rf_classifier = RandomForestClassifier(
    featuresCol='features',
    labelCol='SeriousDlqin2yrs_idx',
    numTrees=100,
    maxDepth=5
)

rf_pipeline = Pipeline(stages=[feature_vector_assembler, label_string_indexer, rf_classifier])
rf_model = rf_pipeline.fit(credit_scoring_train_ddf)
```

#### Gradient Boosted Trees Classifier

```python
from pyspark.ml.classification import GBTClassifier

gbt_classifier = GBTClassifier(featuresCol='features', labelCol='SeriousDlqin2yrs_idx', maxIter=10)
gbt_pipeline = Pipeline(stages=[feature_vector_assembler, label_string_indexer, gbt_classifier])
gbt_model = gbt_pipeline.fit(credit_scoring_train_ddf)
```

#### Logistic Regression Classifier

```python
from pyspark.ml.classification import LogisticRegression

lr_classifier = LogisticRegression(featuresCol='features', labelCol='SeriousDlqin2yrs_idx', maxIter=10)
lr_pipeline = Pipeline(stages=[feature_vector_assembler, label_string_indexer, lr_classifier])
lr_model = lr_pipeline.fit(credit_scoring_train_ddf)
```

### 6. Predictions and Evaluation

After training the models, we make predictions on the test data:

```python
rf_predictions = rf_model.transform(credit_scoring_test_ddf)
gbt_predictions = gbt_model.transform(credit_scoring_test_ddf)
lr_predictions = lr_model.transform(credit_scoring_test_ddf)
```

We evaluate the models based on their ROC curves:

```python
from EvaluationMetrics import bin_classif_eval
rf_eval = bin_classif_eval(rf_predictions)
gbt_eval = bin_classif_eval(gbt_predictions)
lr_eval = bin_classif_eval(lr_predictions)
```

### 7. Conclusion

The model can be tuned further by adjusting hyperparameters and experimenting with different classifiers. Additionally, performance metrics such as AUC, precision, recall, and F1-score should be evaluated on the test set.

---

## Contributing

Feel free to fork this repository, submit issues, or pull requests to improve the model.
