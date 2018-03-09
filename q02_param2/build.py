
# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from greyatomlib.Xgboost_project.q01_myXGBoost.build import myXGBoost

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

param_grid2 = {'gamma': [0, 0.05, 0.1, 0.3, 0.7, 0.9, 1],
               'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1],
               'reg_lambda': [0.05, 0.1, 0.5, 1.0]
               }


# Write your solution here :
def param2(X_train, X_test, y_train, y_test, model, param_grid2):
    return myXGBoost(X_train, X_test, y_train, y_test, model, param_grid2,
                     colsample_bytree=0.7, subsample=0.8, max_depth=2, min_child_weight=4)

