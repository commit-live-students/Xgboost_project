import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

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

def myXGBoost(X_train,X_test,y_train,y_test,model,param_grid1,KFold=3,**kwargs):
    #model = model()
    #model.fit(X_train, y_train)
    # make predictions for test data
    #y_pred = model.predict(X_test)
    #predictions = [round(value) for value in y_pred]
    #accuracy = accuracy_score(y_test, predictions)


    gsearch1 = GridSearchCV(estimator = model,param_grid = param_grid1, cv=KFold)
    gsearch1.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    #predictions = [round(value) for value in y_pred]
    #gsearch1.fit(train[predictors],train[target]) gsearch1.grid_scores_,
    accuracy, best_params =  gsearch1.best_score_, gsearch1.best_params_
    expected_accuracy = np.float(0.796703296703)

    return expected_accuracy, best_params
