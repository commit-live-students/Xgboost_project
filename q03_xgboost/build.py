# %load q03_xgboost/build.py
# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('data/loan_clean_data.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

xgbc = XGBClassifier(seed=9)

def xgboost(X_train, X_test, y_train, y_test, **kwargs):
    best_params1 = {'colsample_bytree': 0.7,
                    'max_depth': 2,
                    'min_child_weight': 4,
                    'subsample': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1.0}
    xgbc.set_params(**best_params1)
    xgbc.fit(X_train, y_train)
    y_pred = xgbc.predict(X_test)
    return accuracy_score(y_test,y_pred)



