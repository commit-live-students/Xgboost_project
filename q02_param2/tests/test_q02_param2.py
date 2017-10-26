from unittest import TestCase
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from inspect import getargspec
from ..build import param2

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


class TestParam2(TestCase):
    def test_param2(self):

        # Input parameters tests
        args = getargspec(param2)
        self.assertEqual(len(args[0]), 5, "Expected argument(s) %d, Given %d" % (5, len(args[0])))
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types

        xgb = XGBClassifier(seed=9)
        accuracy1, best_params1 = param2(X_train, X_test, y_train, xgb, param_grid2)

        self.assertIsInstance(accuracy1, float,
                              "Expected data type for return value is `list`, you are returning %s" % (
                                  type(accuracy1)))

        self.assertIsInstance(best_params1, dict,
                              "Expected data type for return value is `numpy.ndarray`, you are returning %s" % (
                                  type(best_params1)))

        # Return value tests

        expected_best_params = {'reg_alpha': 0, 'reg_lambda': 1.0, 'gamma': 0.9}
        expected_accuracy = 0.796703296703

        self.assertDictEqual(best_params1, expected_best_params, "Expected best_params does not match given best_params")
        self.assertAlmostEqual(accuracy1, expected_accuracy, 4, "Expected accuracy does not match given accuracy")