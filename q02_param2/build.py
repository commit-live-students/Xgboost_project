# %load q02_param2/build.py
# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from  sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
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


def param2(X_train, X_test, y_train, y_test,model,param_grid):

    param_grid1 = {"max_depth": [2, 3, 4, 5, 6, 7, 9, 11],
                   "min_child_weight": [4, 6, 7, 8],
                   "subsample": [0.6, .7, .8, .9, 1],
                   "colsample_bytree": [0.6, .7, .8, .9, 1]
                   }
    ac,bst=myXGBoost(X_train, X_test, y_train, y_test,model,param_grid1,KFold=3)
    m1=model.set_params(**bst)
    h,j=myXGBoost(X_train, X_test, y_train, y_test,m1,param_grid,KFold=3)
    #print model
    return h,j
#model=XGBClassifier(seed=9,max_depth=2,min_child_weight= 4,subsample=.8,colsample_bytree=.7)
#model=XGBClassifier(seed=9)


#h,j=param2(X_train, X_test, y_train, y_test,model,param_grid2)
#print h,j
