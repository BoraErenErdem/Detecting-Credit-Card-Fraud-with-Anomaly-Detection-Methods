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

## 3. Data Preprocessing

In the preprocessing stage, *scaling played a critical role* due to the varying ranges of the features.

Methods Tested: MinMaxScaler, StandardScaler, RobustScaler.

Selected Method: *RobustScaler.*

Rationale: Since the objective of the dataset is credit card fraud detection, the chosen scaling method must be robust to outliers. StandardScaler and MinMaxScaler are sensitive to extreme values, whereas RobustScaler is not. Therefore, RobustScaler was adopted.

<img width="1972" height="765" alt="image" src="https://github.com/user-attachments/assets/0570288c-5cc5-47e2-8727-090ea44c1f26" />

Method Avoided: MinMaxScaler (deemed unsuitable for anomaly detection because it preserves the influence of extreme values).

Skewness Correction: A PowerTransformer (`Yeo-Johnson`) transformation was applied to normalize distributions with high positive and negative skewness, thereby enhancing model stability. Alternative methods were tested, but none yielded significant improvements in balancing skewness across both tails.

<img width="1138" height="1092" alt="image" src="https://github.com/user-attachments/assets/25e37a58-3990-42d8-8a3c-76a241e2533a" />

Additionally, PCA-based two-dimensional visualizations were performed for exploratory purposes; however, the original feature space was retained for model training.

<img width="1566" height="961" alt="image" src="https://github.com/user-attachments/assets/198fc24c-b722-4f2b-8e20-84b506e8675c" />

----

## 4. Model Development

#### 4.1. Isolation Forest

Rationale: An efficient method for high-dimensional and highly imbalanced datasets.

Parameter Setting: The `contamination rate was set to 0.0017`.

Optimization: Hyperparameter tuning was performed using RandomizedSearchCV.

Threshold Selection: The decision threshold was determined based on the Precision-Recall curve, selecting the value that maximized the F1-score (0.0516).

Result: `ROC-AUC = 0.94`, `Recall = 0.49`

<img width="1852" height="1053" alt="image" src="https://github.com/user-attachments/assets/cbb10fa7-0092-4b96-b714-ce988dfbbdf3" />

#### 4.2. PCA Reconstruction Error

Rationale: Fraudulent transactions, being anomalies, cannot be accurately reconstructed compared to normal transactions and therefore yield high reconstruction errors.

Implementation: PCA was applied with `n_components=0.95`, resulting in `24` retained components.

Result: `ROC-AUC = 0.926`, `Recall = 0.72`

<img width="1874" height="1068" alt="image" src="https://github.com/user-attachments/assets/7bfe3313-21bf-4b5f-8aca-2af61e6303fa" />

#### 4.3. Local Outlier Factor (LOF)

Rationale: Density-based methods are suitable for clustered anomalies.

Result: `ROC-AUC = 0.725`, `Recall = 0.05`. Performance was notably poor. This is primarily due to LOF’s reliance on local density approximations, which becomes inefficient in high-dimensional datasets. Consequently, LOF exhibited substantially lower detection performance.

<img width="1854" height="1071" alt="image" src="https://github.com/user-attachments/assets/a9020d8e-4c8b-4f5e-ba40-4b036312c1ad" />

----

## 5. Model Comparison

Given the extreme class imbalance in the dataset, the accuracy metric was considered misleading and therefore excluded. Instead, Recall, Precision, F1-score, ROC-AUC, and Precision-Recall (PR) curves were employed as the primary evaluation metrics.

- **`Isolation Forest:** ROC-AUC = 0.94, Recall = 0.49`**

- **`PCA Reconstruction Error:* ROC-AUC = 0.93, Recall = 0.72`**

- **`LOF: ROC-AUC = 0.73, Recall = 0.05`**


#### PCA Reconstruction Error
##### Why was the PCA ratio set to 0.95? 
Multiple PCA variance retention levels (0.90–0.98) were tested:

- At 0.90, 0.91, 0.92, and 0.93, excessive information loss occurred, leading to poor representation of both normal and fraudulent samples. This increased the number of False Positives.
- At 0.98 and 0.99, over-representation was observed, meaning even anomalous transactions were reconstructed too well, increasing the False Negatives.
- The best trade-off was achieved at ``0.95``, which was therefore selected.

##### Why was the threshold set at 99.5?
The threshold was determined by balancing Recall and F1-score:
##### Threshold = 90.0
- Recall: `0.87` (very high)
- Precision: `0.01` (very low)
- F1-score: `0.03` (very low)
- False Positives: `5612` (5612 people were mistakenly mistaken for fraud)
- Result: Too many false alarms

##### Threshold = 99.8
- Recall: `0.48` (moderate)
- Precision: `0.41` (good)
- F1-score: `0.44` (balanced)
- False Positives: `67` (67 people were mistakenly mistaken for fraud)
- Result: Acceptable, but too many fraud cases missed.


##### Threshold = 99.5 (selected)
- Recall: `0.72` (high)
- Precision: `0.25` (moderate)
- F1-score: `0.37` (reasonably balanced)
- False Positives: `214` (acceptable)
- False Negatives: `27` (low) (very good)

**Conclusion:** The recall metric aims to capture most of the anomalies and is generally more important in anomaly detection. However, in this case, precision—which prevents false predictions—and the F1 score—which ensures a balanced system—remain very low. As a result, although the model is able to detect most anomalies, its low precision causes it to misclassify many normal instances as anomalies, leading to frequent false positives. Therefore, the most reasonable threshold is 99.8, which offers the best balance between the two.
Nevertheless, since the dataset is highly imbalanced and the number of actual fraud cases is very small, the recall parameter becomes more significant. Moreover, the main goal of the model is to capture these rare fraud cases, so I can afford to sacrifice some precision and F1 score. For this reason, I use the threshold at 99.5 (0.56). This way, although there is a tolerable decrease in precision and F1 score, the recall increases to 72%.

<img width="1865" height="1079" alt="image" src="https://github.com/user-attachments/assets/58c3aae9-d6fa-45cc-83b4-44feb150494e" />
<img width="1534" height="935" alt="image" src="https://github.com/user-attachments/assets/34180584-17b5-47f8-8dea-9007f76b65dc" />


#### LOF
##### Why LOF was chosen instead of One-Class SVM?
- One-Class SVM requires subsampling due to computational constraints. However, subsampling leads to significant information loss and higher computational cost.
- One-Class SVM learns only from normal transactions, creating a global decision boundary, which is less interpretable and struggles with highly imbalanced datasets.
- LOF, in contrast, applies a local density-based approach, which is more suitable for detecting rare events such as fraud.
- Based on these considerations, LOF was preferred over One-Class SVM.

<img width="1869" height="1080" alt="image" src="https://github.com/user-attachments/assets/7c751a33-2172-477d-a9c7-0a66589efb91" />
<img width="1859" height="1082" alt="image" src="https://github.com/user-attachments/assets/0d3de978-9160-49e3-9681-5995a36a830c" />

----

#### 5.1. Cost Analysis

Key findings: 
+ PCA Reconstruction Error achieved the highest Recall (`0.72`) and the lowest total cost (`$48,400`), making it highly effective for fraud detection.
+ Isolation Forest achieved the highest ROC-AUC (`0.9407`), but incurred a higher cost than PCA.
+ LOF exhibited poor performance (ROC-AUC = `0.7250`) and was discarded.
+ Ensemble (Isolation Forest + PCA) achieved high ROC-AUC (`0.9423`), but failed to deliver acceptable Recall, Precision, and F1 scores due to difficulties in threshold tuning. It was therefore excluded.

<img width="1698" height="1076" alt="image" src="https://github.com/user-attachments/assets/af819c07-7637-4293-9d17-8e816644e6ec" />
<img width="1868" height="1067" alt="image" src="https://github.com/user-attachments/assets/94423589-a916-4293-b2d5-1e9c65ac3109" />

#### Model Performance Summary
- **Objective**: To compare and visualize the performance of the Isolation Forest, PCA Reconstruction Error, and LOF models, as well as the Ensemble model (Iso+PCA), which was excluded from the project since it failed to detect any fraud cases and performed poorly overall.
- **Metrics**:
  - Isolation Forest: ROC AUC: `0.9407`, F1-Score: `0.3087`, Recall: `0.4898`, Precision: `0.2250`
  - PCA Reconstruction Error: ROC AUC: `0.9256`, F1-Score: `0.37`, Recall: `0.72`, Precision: `0.2491`
  - LOF: ROC AUC: `0.7250`, F1-Score: `0.03`, Recall: `0.05`, Precision: `0.0175`
  - Ensemble (Iso+PCA): ROC AUC: `0.9423`, F1-Score: `0.00`, Recall: `0.00`, Precision: `0.00`
- **Visualizations**:
  - **ROC AUC Bar Plot**: Isolation Forest and PCA show superior AUC values
  - **F1-Score and Recall Bar Plot**:PCA leads in Recall (`0.72`).
  - **ROC Curve and Precision-Recall Curve**: PCA and Isolation Forest outperform LOF, with PCA excelling in Recall.
  - **Confusion Matrix Heatmap**: PCA achieves the lowest False Negatives (`27`).
  - **Cost Bar Plot**: PCA yields the lowest cost (`$48,400`).

**Cost Analysis Summary**:
- PCA: `$48,400` (lowest cost, due to minimized FN).
- Isolation Forest: `$66,500`.
- LOF: `$121,000`.

**Conclusion**: PCA Reconstruction Error was selected as the optimal model, providing the best trade-off between high Recall (`0.72`) and low cost (`$48,400`). LOF was discarded due to poor detection capability, and the Ensemble model was excluded due to failure in fraud detection.

<img width="1871" height="1066" alt="image" src="https://github.com/user-attachments/assets/e0542834-4253-4cb7-9590-d6cc3b5f3a03" />
<img width="1847" height="1066" alt="image" src="https://github.com/user-attachments/assets/ceb01360-ac14-49c6-b88a-e5950e127316" />
<img width="1961" height="703" alt="image" src="https://github.com/user-attachments/assets/7bfe5745-3605-4e1b-bb90-12343b7ac4a0" />

#### False Positive vs. False Negative Cost Assumptions (100, 1000)?
In the context of credit card fraud detection (fraud rate: 0.17%):

- False Negatives (FN) represent undetected fraud, leading to direct financial loss.
- False Positives (FP) cause inconvenience but typically result in lower costs.

##### False Positive (FP) Cost = 100
- Misclassifying a legitimate transaction as fraud.
- Leads to customer inconvenience (transaction declined, extra verification).
- Operational costs for review (customer support, verification systems).
- No direct financial loss. (maybe there may be a loss due to loss of reputation)
- Estimated cost of $100 aligns with values frequently reported in the literature (range: 10–100).


##### False Negative (FN) Cost = 1000
- Misclassifying a fraudulent transaction as legitimate.
- Causes direct financial loss (transaction amount typically 500-2000).
- Additional operational and legal costs (chargebacks, disputes).
- Reputational risk for the bank.
- Literature suggests FN costs are 5–10x higher than FP costs.
- $1000 was selected as a representative value.

<img width="1857" height="1076" alt="image" src="https://github.com/user-attachments/assets/f0ad3baa-c223-4aa4-b0ba-73977e118eb7" />

##### PCA reduced the risk of missing high-cost frauds by minimizing the number of FPs (27). Although the number of FPs (214) was slightly higher, it reduced the total cost due to the low FP cost. With a high recall value, PCA was the most efficient model in both fraud detection and cost.

- FN = `27` (lowest), reducing the risk of costly undetected fraud.

- FP = `214` (moderately high), but FP cost is low.

- Total cost minimized (`$48,400`).

- High Recall (`0.72`) ensures the majority of fraud cases are captured, justifying the selection of PCA as the most efficient model.

----

## 6. Model Interpretability

The correlation analysis conducted after PCA revealed that the variables V14 and V17 exhibited the strongest relationship with fraudulent transactions.

An examination of the anomaly score distributions indicated that fraudulent transactions generally produced higher anomaly scores compared to legitimate transactions.

Within the scope of the cost analysis, the False Positive (FP) to False Negative (FN) ratio was set at 1:10, which is widely regarded as a standard assumption in the financial fraud detection literature.

----

## 7. Conclusion and Recommendations for Improvement

This study demonstrates that credit card fraud can be effectively detected using unsupervised learning-based anomaly detection methods.

#### 7.1. Key Findings

Unsupervised approaches can be effectively applied to datasets with severe class imbalance.

Threshold optimization significantly improved the recall performance of the models.

The PCA Reconstruction Error method stood out in fraud detection due to its high recall value (`0.72`).

#### 7.2. Recommendations for Improvement

`Data Enhancement:` Access to raw, pre-PCA features would improve model interpretability.

`Model Improvement:` A hybrid or ensemble approach combining PCA and Isolation Forest could potentially achieve higher AUC values.

`Hyperparameter Optimization:` Advanced optimization techniques such as Optuna or Bayesian methods can be utilized.

`Hybrid Methods:` Combining supervised and unsupervised learning could yield stronger and more reliable results. Since this study focused solely on unsupervised learning, performance metrics were somewhat limited. (example, creating stacking models with classification algorithms in supervised learning, etc.)

`Advanced Techniques:` Deep learning-based autoencoders or GAN models could further improve detection accuracy.

`Deployment:` The models can be integrated into banking applications for real-time fraud detection.

**Overall Conclusion:** The PCA-based approach emerges as the most suitable method for financial institutions due to its high recall and low cost, while Isolation Forest provides a balanced alternative with consistent performance.
