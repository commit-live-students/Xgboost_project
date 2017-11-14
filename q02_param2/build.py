# %load q02_param2/build.py
# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from greyatomlib.Xgboost_project.q01_myXGBoost.build import myXGBoost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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

# param_grid1 = {'subsample': [0.8], 'colsample_bytree': [0.7], 'max_depth': [2], 'min_child_weight': [4]}

# param_grid = dict(param_grid1.items()+param_grid2.items())


model = XGBClassifier(seed = 9,subsample = 0.8,colsample_bytree = 0.7,max_depth = 2,min_child_weight = 4)





def param2(X_train,X_test,y_train,y_test,model,param_grid):

    model2 = GridSearchCV(model,param_grid)
    model2.fit(X_train,y_train)
    accuracy = accuracy_score(y_test,model2.predict(X_test))
    best_params_temp = model2.best_params_

    your_keys=['reg_alpha','reg_lambda', 'gamma']
    best_params = { your_key: best_params_temp[your_key] for your_key in your_keys }
    return accuracy,best_params

param2(X_train,X_test,y_train,y_test,model,param_grid2)


# Write your solution here :
