# **LOAN PREDICTION PROJECT**

### **Import Packages**
"""

import pandas as pd
import numpy as  np 
import matplotlib.pyplot as plt
import seaborn as sns

"""### **Read and Visualize the data**"""

df= pd.read_csv('train.csv')

df.head()

"""### **Preprocessing the data**
Fill nan values by the median
"""

df = df.fillna(df.median())

"""Encoding Categorical Values"""

df['Loan_Status'] = df.Loan_Status.map({'Y': 1, 'N': 0}).astype(int)

df['Loan_Status']

"""**Although the gender columns seems important when we see the value_counts we see that there is a bias since most of the data correspond to male**"""

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

"""**Refill nan and make all columns numeric**"""

df.fillna(df.median(), inplace=True)
columns = df.columns
for column in columns:
  df[column] = pd.to_numeric(df[column], errors='coerce')

"""**Checking if there is nan values**"""

col_names= df.columns.tolist()
for column in col_names:
  print("Valores Nulos en <{0}>: <{1}>".format(column,df[column].isnull().sum()))

"""**Visualize the Correlation  between the variables**"""

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

"""**Selecting the variables with the higher  absulute value correlation**"""

def correlationdrop(df, sl):
  columns = df.columns
  for column in columns:
      C=abs(df[column].corr(df['Loan_Status']))
      if C < sl:
        df=df.drop(columns=[column])
  return df

df= correlationdrop(df,0.05)

df

"""### **Visualize the data**"""

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

df.CoapplicantIncome[df.Loan_Status==1].plot(kind='kde')
plt.title('Loan accepted by Coaplicant Income')
plt.show()

"""## **Model**

**Choose the dependent and independent columns**
"""

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

"""**Scaling the data**"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X= sc.fit_transform(x)

"""**Split the data on the train and test datasets**"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state= 0)

"""### **Support Vector Classifier**"""

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma= 0.2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""**Cross Validation**"""

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

"""**Grid Seach**"""

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
