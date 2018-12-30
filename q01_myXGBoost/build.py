# %load q01_myXGBoost/build.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

xgbc = XGBClassifier(seed=9)

def myXGBoost(X_train, X_test, y_train, y_test, model, param_grid, KFold=3, **kwargs):
    grid_model = GridSearchCV(model, param_grid, cv=KFold)
    grid_model.fit(X_train, y_train)
    y_pred = grid_model.predict(X_test)
    return accuracy_score(y_test, y_pred), grid_model.best_params_



