#Multi Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data
dataset = pd.read_csv('50_Startups.csv')
dataset.head()
y = dataset.iloc[:, -1]
X = dataset.iloc[:, 0:4]
y.shape

#preprocessing
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
state = labelencoder.fit_transform(dataset.iloc[: , 3])
X.iloc[:, -1] = state
X.head()

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
X
#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
model.coef_

#Optimizing the Model
import statsmodels.formula.api as sm
X_opt = X[: , [0,1,2,3,4]]
OptimizedModel = sm.OLS(endog=y, exog=X_opt).fit()
OptimizedModel.summary()
