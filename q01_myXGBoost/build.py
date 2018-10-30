# %load q01_myXGBoost/build.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.model_selection import cross_val_score

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

param_grid1 = {'max_depth': [2, 3, 4, 5, 6, 7, 9, 11],
               'min_child_weight': [4, 6, 7, 8],
               'subsample': [0.6, .7, .8, .9, 1],
               'colsample_bytree': [0.6, .7, .8, .9, 1]
               }
xgb = XGBClassifier(seed=9)

# Write your solution here :
def myXGBoost(X_train, X_test, y_train, y_test, model, param_grid, KFold=3,**kwargs):
    
    for i, j in kwargs.items():
        lst1=list()
        lst1.append(j)
        param_grid[i]=lst1
        
    acc_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, param_grid, scoring=acc_scorer,cv=KFold)
    grid_obj = grid_obj.fit(X_train, y_train)
    y_pred=grid_obj.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    best_params=grid_obj.best_params_
    return accuracy, best_params

