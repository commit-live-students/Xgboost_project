# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

# Write your solution here :
def xgboost (X_train, X_test, y_train, y_test,**kwargs):
    xgb = XGBClassifier()
    param_grid = {"max_depth": [ 2],
               "min_child_weight": [ 8],
               "subsample": [0.6, .7, .8, .9],
               "colsample_bytree": [0.6, .7, .8, .9, 1]
               }
    grid = GridSearchCV(xgb,param_grid,cv = 5)
    grid.fit(X_train,y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy.item()
