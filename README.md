# Credit Card Fraud Is Used By Identifying and Using Anomaly Detection Methods

## 1. Introduction

This study was conducted by Bora Eren ERDEM.

Credit card fraud is a critical problem in the financial sector, causing annual losses amounting to billions of dollars. What fundamental factors distinguish fraudulent transactions from legitimate ones? How can such rare events be detected in highly imbalanced datasets, where fraudulent transactions account for less than 0.2% of all records?

To address these questions, the Credit Card Fraud Detection dataset (available on Kaggle, with features anonymized through PCA) was employed. The primary objective of this study is to detect fraudulent transactions using unsupervised machine learning methods. This approach is particularly relevant since, in real-world scenarios, fraud data is often scarcely labeled, and new fraud patterns emerge unpredictably.

You can find the dataset I used here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**In this study, fraudulent transactions were treated as “anomalies,” and the following methods were applied:**

- Isolation Forest

- PCA Reconstruction Error

- Local Outlier Factor (LOF)


**The project was structured in the following stages:**

- Data Exploration and Preliminary Analysis

- Data Preprocessing

- Model Development

- Model Evaluation and Comparison

- Model Interpretability and Cost Analysis

- Results and Recommendations

----

## 2. Dataset

The dataset consists of *284,807 transactions* and *31 features*.

*V1–V28:* Features anonymized through PCA transformation, representing transaction-related details.

*Time:* The elapsed time in seconds since the first transaction.

*Amount:* Transaction amount (USD).

*Class:* Binary target variable (0: normal, 1: fraud).

**There are a total of 492 fraudulent transactions, corresponding to approximately 0.172% of the dataset. This extreme class imbalance reflects a realistic scenario commonly encountered in financial fraud detection.**

#### 2.1. Preliminary Analysis Findings

The fraud rate was calculated as `y.mean() ≈ 0.0017`, which was used as the basis for setting the contamination parameter in the models

Distribution analysis revealed that the Amount variable is highly skewed.

<img width="1878" height="712" alt="Ekran görüntüsü 2025-08-14 181134" src="https://github.com/user-attachments/assets/34d36d7d-e818-47f1-8da0-86250bcf597e" />

After reducing the data to two dimensions via PCA, fraudulent transactions appeared as outliers, separated from the main cluster of legitimate transactions.

#### 2.2. Decisions

`StratifiedShuffleSplit:` A StratifiedShuffleSplit with an 80/20 ratio was applied to preserve class imbalance. *The reason for using StratifiedShuffleSplit instead of a simple train-test split is that anomalies represent a very small and imbalanced portion of the dataset. Stratified sampling ensures the preservation of this distribution, leading to more reliable anomaly detection results compared to a random split.*

<img width="1433" height="1044" alt="Ekran görüntüsü 2025-08-14 181231" src="https://github.com/user-attachments/assets/b12a536a-4815-46da-895d-f813072b6c3e" />

**`No SMOTE or similar methods were applied`: In the financial domain, generating synthetic fraudulent transactions risks introducing unrealistic patterns, and therefore was deliberately avoided.**

`Outlier Removal:` A Z-score threshold (>3 standard deviations) was applied to eliminate extreme values.

<img width="1205" height="857" alt="image" src="https://github.com/user-attachments/assets/ddfc488c-7e29-4411-8674-1a7bc4dadc1f" />

----
