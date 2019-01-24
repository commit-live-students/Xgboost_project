# %load q01_myXGBoost/build.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

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
acc_scorer = make_scorer(accuracy_score)
classifier = XGBClassifier()

def myXGBoost(X_train, X_test,y_train,y_test,classifier,param_grid,KFold=3):
    
    grid = GridSearchCV(classifier, param_grid1, scoring=acc_scorer)
    grid=grid.fit(X_train,y_train)
    y_test1=grid.predict(X_test)
    accuracy_clf_1 = accuracy_score(y_test, y_test1)
    grid.grid_scores_
    return(accuracy_clf_1,grid.best_params_)

myXGBoost(X_train, X_test,y_train,y_test,classifier,param_grid1,KFold=3)


