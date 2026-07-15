# Fraud_Detection
A machine learning project that detects fraudulent financial transactions by combining data from multiple sources, engineering behavioral features, and comparing Random Forest, Logistic Regression, and XGBoost classifiers with SHAP used to explain the winning model's predictions.

## Business Problem
Financial institutions lose significant revenue to fraudulent transactions, and fraud is inherently rare compared to legitimate activity, which makes it a hard needle-in-a-haystack problem for standard classification models. 
This project builds and compares several models to flag suspicious transactions, prioritizing the ability to catch as much fraud as possible (recall) while keeping the analysis grounded in real customer, merchant, and account behavior data.

## Dataset
The project merges 10 separate CSV sources (transaction records, amounts, customer profiles, merchant data, account activity, anomaly scores, and fraud/suspicious activity flags) into a single dataset joined on `TransactionID`, `CustomerID`, and `MerchantID`.

## Approach
1. **Data merging & cleaning**: combined 10 raw tables, checked for nulls and duplicates, and reviewed data types.
2. **Exploratory data analysis**: correlation heatmaps, fraud vs. legitimate transaction distribution, boxplots of numerical features, and customer age profiling.
3. **Feature engineering**: created behavioral and time-based features:
   - `Amount_Diff` (difference between transaction amount and account amount)
   - `TransactionFrequency` per customer
   - Cyclical hour and weekday encodings (`hour_sin`/`hour_cos`, `day_sin`/`day_cos`)
   - `Amount_to_Avg_Ratio` (spend relative to a customer's historical average)
4. **Encoding & feature selection**: simplified transaction category into Online vs. Other, and dropped identifier/non-predictive columns (names, addresses, IDs, timestamps).
5. **Train/test split & scaling**: 80/20 stratified split, with `StandardScaler` fit on training data only.
6. **Class imbalance handling**: applied **SMOTE** to the training set to oversample the minority (fraud) class.
7. **Modeling**: trained and tuned three classifiers:
   - Random Forest (with a custom probability threshold of 0.20 to increase fraud sensitivity)
   - Logistic Regression (tuned via `GridSearchCV` over regularization strength and solver)
   - XGBoost
8. **Evaluation**: classification reports, confusion matrices, and ROC/AUC curves for each model.
9. **Model explainability**: SHAP `TreeExplainer` used on the champion model to identify which features drive fraud predictions.

## Results
| Model | ROC AUC |
|---|---|
| **Random Forest (Champion)** | **0.746** |
| XGBoost | 0.730 |
| Logistic Regression (tuned) | 0.662 |

Random Forest was selected as the champion model based on ROC AUC. Given the severe class imbalance (only 9 fraud cases in the 200-row test set), recall on the fraud class was prioritized over overall accuracy — the Random Forest model at a 0.20 threshold caught 89% of fraud cases in the test set, at the cost of a higher false-positive rate.

*Note: with such a small number of fraud cases in the test set, these metrics should be treated as directional rather than statistically robust — a larger fraud sample would give a more reliable performance estimate.*

## Tech Stack
- **Python**: pandas, numpy
- **Modeling**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Visualization**: matplotlib, seaborn
- **Explainability**: SHAP
- **Environment**: Google Colab

## Repository Structure
├── ODL_Group.ipynb        # Full analysis notebook (EDA, feature engineering, modeling, evaluation)
└── README.md               # Project documentation


*Note: raw CSV data files are not included in this repository. To run the notebook, place `account_activity.csv`, `amount_data.csv`, `anomaly_scores.csv`, `customer_data.csv`, `fraud_indicators.csv`, `merchant_data.csv`, `suspicious_activity.csv`, `transaction_category_labels.csv`, `transaction_metadata.csv`, and `transaction_records.csv` in the same directory as the notebook.*

## How to Run
1. Clone this repository
2. Open `ODL_Group.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab
3. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib shap joblib`
4. Add the required CSV files to the working directory
5. Run all cells
