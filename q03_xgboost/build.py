# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)


# Write your solution here :
# exercise -3
def xgboost(X_train, X_test, y_train, y_test, **kwargs):
    kwargs['random_state'] = 9
    xgb = XGBClassifier(**kwargs)
    xgb.fit(X_train, y_train)
    y_pred_test = xgb.predict(X_test)
    return accuracy_score(y_true=y_test, y_pred=y_pred_test)
