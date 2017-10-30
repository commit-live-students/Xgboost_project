from unittest import TestCase
from inspect import getargspec
from ..build import myXGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv('data/loan_clean_data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

param_grid1 = {"max_depth": [2, 3, 4, 5, 6, 7, 9, 11],
               "min_child_weight": [4, 6, 7, 8],
               "subsample": [0.6, .7, .8, .9, 1],
               "colsample_bytree": [0.6, .7, .8, .9, 1]
               }


class TestMyXGBoost(TestCase):
    def test_myXGBoost(self):

        # Input parameters tests
        args = getargspec(myXGBoost)
        self.assertEqual(len(args[0]), 6, "Expected argument(s) %d, Given %d" % (6, len(args[0])))
        self.assertEqual(args[3], (3,), "Expected default values do not match given default values")

        # Return data types

        xgb = XGBClassifier(seed=9)
        accuracy, best_params = myXGBoost(X_train, X_test, y_train, xgb, param_grid1, 3)

        self.assertIsInstance(accuracy, float,
                              "Expected data type for return value is `list`, you are returning %s" % (
                                  type(accuracy)))

        self.assertIsInstance(best_params, dict,
                              "Expected data type for return value is `numpy.ndarray`, you are returning %s" % (
                                  type(best_params)))

        # Return value tests
        expected_best_params = {'subsample': 0.8, 'colsample_bytree': 0.7, 'max_depth': 2, 'min_child_weight': 4}
        expected_accuracy = 0.796703296703

        self.assertDictEqual(best_params, expected_best_params, "Expected best_params does not match given best_params")
        self.assertAlmostEqual(accuracy, expected_accuracy, 4, "Expected accuracy does not match given accuracy")
