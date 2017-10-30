from unittest import TestCase
from sklearn.model_selection import train_test_split
from ..build import xgboost
from inspect import getargspec
import pandas as pd

# load data
dataset = pd.read_csv('data/loan_clean_data.csv')
# split data into X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)


class TestXgboost(TestCase):
    def test_xgboost(self):

        # Input parameters tests
        args = getargspec(xgboost)
        self.assertEqual(len(args[0]), 3, "Expected argument(s) %d, Given %d" % (3, len(args[0])))
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types

        accuracy = xgboost(X_train, X_test, y_train, colsample_bytree=0.7, gamma=0.9, max_depth=2, min_child_weight=6,
                      subsample=1, n_estimators=100, reg_alpha=0, reg_lambda=0.05, learning_rate=0.1)

        self.assertIsInstance(accuracy, float,
                              "Expected data type for return value is `list`, you are returning %s" % (
                                  type(accuracy)))

        # Return value tests
        expected_accuracy = 0.802197802198
        self.assertAlmostEqual(accuracy, expected_accuracy, 3, "Expected accuracy does not match given accuracy")

