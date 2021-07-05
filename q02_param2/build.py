# %load q02_param2/build.py
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
    param_grid1 = {'max_depth': [2, 3, 4, 5, 6, 7, 9, 11],
               'min_child_weight': [4, 6, 7, 8],
               'subsample': [0.6, .7, .8, .9, 1],
               'colsample_bytree': [0.6, .7, .8, .9, 1]
               }
    accuracy,best_params=myXGBoost(X_train, X_test, y_train, y_test,model,param_grid1,KFold=3)
    
    model1=model.set_params(**best_params)
    accuracy1,best_params1=myXGBoost(X_train, X_test, y_train, y_test,model1,param_grid2,KFold=3)
    
    # print (accuracy1, best_params1)    
    
    return accuracy1,best_params1

# param2(X_train, X_test, y_train, y_test, XGBClassifier(seed=9), param_grid2)



