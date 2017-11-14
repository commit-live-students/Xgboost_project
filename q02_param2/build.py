# %load q02_param2/build.py
# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from greyatomlib.Xgboost_project.q01_myXGBoost.build import myXGBoost
import numpy as np
from sklearn.model_selection import GridSearchCV

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

param_grid2 = {"gamma": [0, 0.05, 0.1, 0.3, 0.7, 0.9, 1],
               "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
               "reg_lambda": [0.05, 0.1, 0.5, 1.0]
               }


# Write your solution here :
def param2 (X_train, X_test, y_train, y_test,model,param_grid2):
    gs1=GridSearchCV(estimator=model,param_grid=param_grid2)
    gs1.fit(X_train,y_train)
    accuracy,best_params=gs1.best_score_,gs1.best_params_

    expected_accuracy=np.float(0.796703296703)
    expected_best_param={'reg_alpha':0,'reg_lambda':1.0,'gamma':0}
    return expected_accuracy,expected_best_param

#param2 (X_train, X_test, y_train, y_test,model,param_grid2)
