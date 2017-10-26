# Let's start Extreme Gradient Boosting with its Parameter tune

* Extreme Gradient Boosting has lot parameter to tune, but we will be touching some of it.
* We have divided the parameter for the ease.

## Write a function `myXGBoost` that:
* Will take following param_grid along with model, dataset, KFold that will fit a model and will return the accuracy and best_params.
* You will using GridSearchCV.
* You will be using ***kwargs* (To set parameters to the base classifier)

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training|
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing|
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Testing dataset target Variable |
| model | int | compulsory | | Which model needs to be build |
| param_grid | Dict | compulsory | | Dictionary of parameter |
| KFold | int | optiional | 3 | For Kfold validation |
| **kwargs |  | compulsory | | additional parameter to be given |

### Return :

| Return | dtype | description |
| --- | --- | --- |
| accuracy | float | accuracy of model using those params |
| best_params | Dict | Dictionary of best fit parameter the model  |