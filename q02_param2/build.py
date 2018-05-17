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

param_grid2 = {"gamma": [0, 0.05, 0.1, 0.3, 0.7, 0.9, 1],
               "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
               "reg_lambda": [0.05, 0.1, 0.5, 1.0]
               }


# Write your solution here :
def param2(X_train, X_test, y_train, y_test, model, param_grid):
    param_grid1 = {"max_depth": [2, 3, 4, 5, 6, 7, 9, 11],
               "min_child_weight": [4, 6, 7, 8],
               "subsample": [0.6, .7, .8, .9, 1],
               "colsample_bytree": [0.6, .7, .8, .9, 1]
               }
    acc_score, best_params = myXGBoost(X_train, X_test, y_train, y_test, model=model, param_grid=param_grid1, KFold=3)
    # Get the best parameters from iteration-1
    # Append the best parameters and the new parameters to create
    # new set of parameters
    param_grid3 = { key_: [val_] for key_, val_ in best_params.items() }
    param_grid3.update(param_grid)
    # Use previous function with new set of parameters (iteration-1 best params and new params param-grid)
    acc_score1, best_params1 = myXGBoost(X_train, X_test, y_train, y_test, model=model, param_grid=param_grid3)
    # Return only specific_params that were passed as part of param_grid in dictionary
    specific_params = {}
    for key_, value_ in best_params1.items():
        if key_ not in best_params.keys():
            specific_params[key_] = best_params1[key_]
    return acc_score1, specific_params
