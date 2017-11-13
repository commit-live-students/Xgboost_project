
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

param_grid2 = {"gamma": [0, 0.05, 0.1, 0.3, 0.7, 0.9, 1],
               "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
               "reg_lambda": [0.05, 0.1, 0.5, 1.0]
               }

model = XGBClassifier(seed=9)

# Write your solution here :
def param2(X_train, X_test, y_train, y_test,model,param_grid):

    param_grid1 = {"max_depth": [2, 3, 4, 5, 6, 7, 9, 11],
               "min_child_weight": [4, 6, 7, 8],
               "subsample": [0.6, .7, .8, .9, 1],
               "colsample_bytree": [0.6, .7, .8, .9, 1]
               }
    cc_score,bestParam = myXGBoost(X_train, X_test, y_train, y_test,model,param_grid1,3)
    #print(bestParam)
    #dic ={}

    #updatePara = {'subsample': 0.8, 'colsample_bytree': 0.7, 'max_depth': 2, 'min_child_weight': 4}

    param_g={}
    for k, v in bestParam.items():
        #print(k,v)
        param_g[k]=[v]

    #print(param_g)

    #updateParam = {'subsample': [0.8], 'colsample_bytree': [0.7], 'max_depth': [2], 'min_child_weight': [4]}

    updateParam=param_g.copy()
    updateParam.update(param_grid)
    #print(updateParam)
    cc_score2,bestParam2= myXGBoost(X_train, X_test, y_train, y_test,model,updateParam,3)

    #print(cc_score2,bestParam2)
    update_best_param={k: v for k, v in bestParam2.items() if k not in param_g}
    return (cc_score2.item(),update_best_param)


# accuracy1, best_params1 =param2(X_train, X_test, y_train, y_test,model,param_grid2)
# print(type(accuracy1))
# print( type(best_params1))
# print(accuracy1)
# print(best_params1)
