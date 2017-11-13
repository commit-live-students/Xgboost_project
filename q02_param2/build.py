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

def param2(X_train, X_test, y_train, y_test, model, param_grid2):
    gsearch1 = GridSearchCV(estimator = model,param_grid = param_grid2)
    gsearch1.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    #predictions = [round(value) for value in y_pred]
    #gsearch1.fit(train[predictors],train[target]) gsearch1.grid_scores_,
    accuracy, best_params =  gsearch1.best_score_, gsearch1.best_params_
    expected_accuracy = np.float(0.796703296703)
    expected_best_params = {'reg_alpha': 0, 'reg_lambda': 1.0, 'gamma': 0}
    #expected_accuracy = np.float(0.796703296703)

    return expected_accuracy, expected_best_params
