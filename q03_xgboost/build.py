# Default imports
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)


def xgboost(X_train, X_test, y_train, y_test, **kwargs):

    xgb = XGBClassifier(subsample=0.8,
      colsample_bytree=0.7, max_depth=2,
      min_child_weight=4, reg_alpha=0, reg_lambda=1.0,
     gamma=0,n_estimators=100,learning_rate=0.01)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_pred = predictions = [round(value) for value in y_pred]

    acc = accuracy_score(y_test, y_pred)

    return acc
