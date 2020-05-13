# -*- coding: utf-8 -*-
"""Loan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TBYbn7eWCTMfuq0hzzNVmiVp4-W4OZTP
"""

import pandas as pd
import numpy as  np 
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('train.csv')

df.head()

df = df.fillna(df.median())

df['Loan_Status'] = df.Loan_Status.map({'Y': 1, 'N': 0}).astype(int)

df['Loan_Status']

grid = sns.FacetGrid(df, col='Loan_Status')
grid.map(plt.hist, 'Gender')

df['Gender'].value_counts()

df= df.replace({"Gender":{"Male":1, "Female":0 }})
df =df.replace({"Married" :{"Yes":1, "No":0}})

df['Dependents'] = df['Dependents'].replace('3+', '3')

df['Dependents']=pd.to_numeric(df['Dependents'], errors='coerce')

df

df['Self_Employed'].value_counts()

df= df.replace({"Self_Employed":{"Yes":1, "No":0 }})

df['Education'].value_counts()

df= df.replace({"Education":{"Graduate":1, "Not Graduate":0 }})

df

df = df.drop(columns=['Loan_ID'])

df['Property_Area'].value_counts()

df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})

df

df.describe()

columns = df.columns
for column in columns:
  df[column] = pd.to_numeric(df[column], errors='coerce')
df.dropna(inplace=True)

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

def correlationdrop(df, sl):
  columns = df.columns
  for column in columns:
      C=abs(df[column].corr(df['Loan_Status']))
      if C < sl:
        df=df.drop(columns=[column])
  return df

df= correlationdrop(df,0.05)

df

df.groupby(['Education','Property_Area'])['Loan_Status'].count()

df.groupby(['Education','Married'])['Loan_Status'].count()

df.Property_Area[df.Loan_Status==1].value_counts(normalize = True).plot(kind='bar', alpha = 0.5)
plt.title('Loan Accepted by Property Area')
plt.show()

df.Credit_History[df.Loan_Status==1].value_counts(normalize = True).plot(kind='bar', alpha = 0.5)
plt.title('Loan Accepted by Credit Hystory')
plt.show()

df.Married[df.Loan_Status==1].value_counts(normalize = True).plot(kind = 'bar')
plt.title('Loan Accepted by Maritage State')
plt.show()

df.LoanAmount[df.Loan_Status==1].plot(kind='kde')
plt.title('Loan accepted by amount')
plt.show()

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X= sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

