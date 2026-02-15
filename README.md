Machine Learning Model Comparison Study

a. Problem Statement
The objective of this project is to perform a comparative analysis of multiple machine learning classification models on a dataset.

The goal is to evaluate how different algorithms perform in terms of predictive accuracy and robustness using standard evaluation metrics such as Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).
This comparison helps in identifying the most suitable model for the chosen dataset and understanding the trade-offs between different learning approaches.

b. Dataset Description
The dataset used in this study is a supervised classification dataset consisting of multiple input features and a categorical target variable.
Dataset: Heart Disease UCI
Source: Kaggle
Download Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Key characteristics of the dataset:

Contains both numerical and categorical features
Target variable represents the presence or absence of heart disease (class labels)
Preprocessing steps applied:
Handling missing values (if any)
Feature scaling where required
Train–test split to ensure unbiased model evaluation
The dataset was used consistently across all models to maintain a fair and reliable performance comparison.


Project Structure-
Machine-Learning-Model-Comparison-Study/
│
├── model/                          # Folder ( contains trained models)
│
├── .DS_Store                       # Mac system file (can be ignored)
│
├── README.md                       # Project documentation
│
├── app.py                          # Streamlit web application
│
├── heart_disease_uci.csv          # Dataset file
│
└── requirements.txt  

c. Models Used 
The following six machine learning models were implemented and evaluated:
Logistic Regression
Decision Tree
k-Nearest Neighbors (kNN)
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)
Each model was trained on the same training data and evaluated using identical performance metrics.

Model Evaluation Metrics Comparison Table

| **ML Model Name**   | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1 Score** | **MCC** |
| ------------------- | ------------ | ------- | ------------- | ---------- | ------------ | ------- |
| Logistic Regression | 0.6066       | 0.8287  | 0.5617        | 0.6066     | 0.5821       | 0.3637  |
| Decision Tree       | 0.4426       | 0.6683  | 0.4474        | 0.4426     | 0.4392       | 0.1609  |
| kNN                 | 0.5246       | 0.7686  | 0.4419        | 0.5246     | 0.4797       | 0.2091  |
| Naive Bayes         | 0.5246       | 0.8389  | 0.5491        | 0.5246     | 0.5348       | 0.2829  |
| Random Forest       | 0.5738       | 0.8256  | 0.4774        | 0.5738     | 0.5199       | 0.2826  |
| XGBoost             | 0.5246       | 0.7859  | 0.5281        | 0.5246     | 0.5227       | 0.2601  |


Observations on Model Performance 

| **ML Model Name**   | **Observation about Model Performance**                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved the highest overall accuracy and MCC, indicating strong and stable performance. It also showed a high AUC, making it a reliable baseline model for this dataset. |
| Decision Tree       | Performed poorly compared to other models, with the lowest accuracy and MCC. This suggests overfitting or sensitivity to data variations.                                 |
| kNN                 | Delivered moderate performance. While recall was reasonable, lower precision and MCC indicate limited discriminative power for this dataset.                              |
| Naive Bayes         | Showed a strong AUC value, indicating good class separation. However, overall accuracy was moderate due to its strong independence assumptions.                           |
| Random Forest       | Improved performance over the single decision tree, showing better accuracy and balanced metrics. Ensemble learning helped reduce overfitting.                            |
| XGBoost             | Provided competitive and consistent results across metrics. Although not the top performer in accuracy, it demonstrated stable and robust classification behavior.        |



Conclusion
From the comparative study, Logistic Regression emerged as the best-performing model in terms of overall accuracy and MCC, while ensemble methods such as Random Forest and XGBoost demonstrated stable and reliable performance.
This study highlights the importance of evaluating multiple metrics rather than relying on accuracy alone when selecting a classification model.


