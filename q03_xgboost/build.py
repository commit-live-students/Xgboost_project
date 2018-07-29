# %load q03_xgboost/build.py
# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer
import numpy as np

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)


# Write your solution here :
def xgboost(X_train, X_test, y_train, y_test,**kwargs):
    dict1=dict()
    for i, j in kwargs.items():
        lst1=list()
        lst1.append(j)
        dict1[i]=lst1
    
    xgb = XGBClassifier(seed=9)
    acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(xgb, dict1, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)
    y_pred=grid_obj.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    return accuracy

