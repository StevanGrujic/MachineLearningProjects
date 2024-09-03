### Project 1 Overview

This project focuses on predicting whether a client will subscribe to a term deposit using a bank marketing dataset. The analysis involved preprocessing, feature selection, and building various machine learning models to achieve accurate predictions.

### Key Steps

- **Data Preprocessing**: Handled missing values, encoded categorical variables, and scaled features using MinMaxScaler.
- **Feature Selection**: Utilized techniques like Chi-squared tests, RFE, and PCA to identify and retain the most important features.
- **Model Building**: Trained and evaluated models including Logistic Regression, Decision Tree, Naive Bayes, Random Forest, AdaBoost, and Bagging Classifier.
- **Model Evaluation**: Focused on metrics like F1 Score, ROC-AUC, and Precision due to the imbalanced nature of the dataset.

### Conclusion for project 1

The project provided insights into handling imbalanced datasets and demonstrated the effective use of machine learning techniques to predict client behavior in a banking context.


### Project 2 Overview

This project analyzes airline passenger satisfaction using a dataset that includes various features related to passenger demographics, flight details, and service satisfaction. The goal is to build machine learning models to predict whether a passenger is satisfied based on these features.

### Key Steps

- **Data Preprocessing**:
  - Handled missing values and dropped irrelevant features.
  - Encoded categorical variables using techniques like label encoding and one-hot encoding.
  - Scaled features using MinMaxScaler and StandardScaler.

- **Exploratory Data Analysis (EDA)**:
  - Conducted a thorough EDA to understand feature distributions and correlations.
  - Visualized satisfaction levels across different categories such as gender, travel class, and type of travel.

- **Feature Engineering**:
  - Identified and removed highly correlated features.
  - Considered dimensionality reduction using PCA due to high correlation among some attributes.

- **Model Building**:
  - Developed multiple models, including Artificial Neural Networks (ANN) with and without Dropout layers.
  - Used early stopping and various optimizers like Adam to enhance model performance.

- **Model Evaluation**:
  - Evaluated models based on accuracy, precision, recall, F1-score, and confusion matrix.
  - Addressed overfitting by introducing Dropout layers, improving model generalization.

### Conclusion for project 2

The project successfully applied machine learning techniques to predict airline passenger satisfaction with high accuracy. The introduction of Dropout in ANN models effectively mitigated overfitting, leading to a robust predictive model.

### Project 3 Overview

This project focuses on customer segmentation using a dataset that includes customer demographics, product purchases, and promotional responses. The primary goal is to identify distinct customer segments through clustering algorithms, allowing the company to tailor its marketing strategies effectively.

### Data Preparation

- **Data Cleaning**:
  - Handled missing values and duplicates.
  - Encoded categorical variables for model compatibility.

- **Feature Engineering**:
  - Created new features such as `Living_With`, `Spent`, `Children`, `Is_Parent`, and `Family_Size` to enhance the segmentation model.
  - Performed normalization and dimensionality reduction on the dataset to improve clustering performance.

- **Data Visualization**:
  - Visualized key features and their distributions to understand the customer base better.
  - Used plots to explore relationships between different variables, aiding in feature selection.

### Clustering Models

Multiple clustering algorithms were tested, including K-Means and Agglomerative Clustering, across various data preprocessing scenarios:

- **K-Means Clustering**:
  - Tested on normalized and standardized datasets with and without dimensionality reduction.
  - The best result was achieved with normalized and reduced data, yielding a Silhouette score of 0.597 and a Calinski-Harabasz score of 2367.

- **Agglomerative Clustering**:
  - Also tested across different preprocessed datasets.
  - Generally performed well but did not surpass the results obtained from K-Means on normalized and reduced data.

### Results and Conclusion for Project 3

- **Best Model**:
  - The most effective model was K-Means applied to the normalized and reduced dataset, using features such as `Income`, `DealsPurchases`, `WebPurchases`, `CatalogPurchases`, `StorePurchases`, `WebVisitsMonth`, `Living_With`, `Spent`, `Children`, `Is_Parent`, and `Family_Size`.
  - This model achieved the highest Silhouette score of 0.597 and a Calinski-Harabasz score of 2367, making it the most suitable for customer segmentation in this context.
