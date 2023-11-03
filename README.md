# Loan-Eligibility-prediction
# Loan Eligibility Prediction using Machine Learning Models in Python

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Loading Dataset](#loading-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Introduction
Have you ever thought about the apps that can predict whether you will get your loan approved or not? In this article, we are going to develop one such model that can predict whether a person will get his/her loan approved or not by using some of the background information of the applicant like the applicant’s gender, marital status, income, etc.

## Importing Libraries
In this step, we will be importing libraries like NumPy, Pandas, Matplotlib, etc.
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

Loading Dataset

 

 

df = pd.read_csv('loan_data.csv')
df.head()

# Exploratory Data Analysis (EDA)

EDA refers to the detailed analysis of the dataset which uses plots like distplot, barplots, etc.
Piechart for LoanStatus

 

temp = df['Loan_Status'].value_counts()
plt.pie(temp.values,
        labels=temp.index,
        autopct='%1.1f%%')
plt.show()

# Countplots for Gender and Marital Status

 

plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=df, x=col, hue='Loan_Status')
plt.tight_layout()
plt.show()

# Distribution Plots for ApplicantIncome and LoanAmount

 

plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Boxplots to Identify Outliers

 

plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# Remove Outliers

 

df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400000]

Loan Amount Analysis

 

df.groupby('Gender').mean()['LoanAmount']

Loan Amount Analysis by Marital Status and Gender

 

df.groupby(['Married', 'Gender']).mean()['LoanAmount']

Data Preprocessing

In this step, we will split the data for training and testing and preprocess the training data.

 

# Code for data preprocessing

Model Development

We will use Support Vector Classifier for training the model.

 

# Code for model development

Model Evaluation

Model evaluation can be done using the confusion matrix.

 from sklearn.metrics import roc_auc_score
model = SVC(kernel='rbf')
model.fit(X, Y)

print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))
print()
Training Accuracy :  0.6136363636363635
Validation Accuracy :  0.4908403026682596

# Code for model evaluation
plt.figure(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

![image](https://github.com/surajmhulke/Loan-Eligibility-prediction/assets/136318267/054a5531-c1cc-4da5-b39d-995983d251e1)
from sklearn.metrics import classification_report
print(classification_report(Y_val, model.predict(X_val)))
          precision    recall  f1-score   support

           0       0.26      0.29      0.28        31
           1       0.72      0.69      0.70        81

    accuracy                           0.58       112
   macro avg       0.49      0.49      0.49       112
weighted avg       0.59      0.58      0.59       112


# Conclusion

As this dataset contains fewer features, the performance of the model is not up to the mark. Maybe with a better and bigger dataset, we will be able to achieve better accuracy.
