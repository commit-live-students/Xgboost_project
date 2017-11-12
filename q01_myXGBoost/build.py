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

param_grid1 = {"max_depth": [2, 3, 4, 5, 6, 7, 9, 11],
               "min_child_weight": [4, 6, 7, 8],
               "subsample": [0.6, .7, .8, .9, 1],
               "colsample_bytree": [0.6, .7, .8, .9, 1]
               }


# Write your solution here :
def myXGBoost(X_train, X_test, y_train, y_test, model, param_grid, Kfold=3, **kwargs):
    gridsearch=GridSearchCV(estimator=model, param_grid=param_grid, cv=Kfold, return_train_score=True)
    gridsearch.fit(X_train,y_train)
    best_params = gridsearch.best_params_
    score = accuracy_score(y_test, gridsearch.predict(X_test))
    model.set_params(**best_params)
    return score, best_params
    
