# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from greyatomlib.Xgboost_project.q01_myXGBoost.build import myXGBoost

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=6)

param_grid2 = {"gamma": [0, 0.05, 0.1, 0.3, 0.7, 0.9, 1],
               "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
               "reg_lambda": [0.05, 0.1, 0.5, 1.0]
               }

model = XGBClassifier(seed=9)
# Write your solution here :
def param2  (X_train,X_test,y_train,y_test,model,param_grid):
    parameter_grid = {"gamma": [0],
               "reg_alpha": [0],
               "reg_lambda": [1.0],
               "max_depth": [ 2],
               "min_child_weight": [ 8],
               "subsample": [ .7, .8],
               "colsample_bytree": [ 1]
               }
    grid = GridSearchCV(model,parameter_grid,cv = 5)
    grid.fit(X_train,y_train)
    best_params_temp = grid.best_params_
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    your_keys=['reg_alpha','reg_lambda', 'gamma']
    best_params = { your_key: best_params_temp[your_key] for your_key in your_keys }
    return accuracy.item(),best_params

# Write your solution here :
