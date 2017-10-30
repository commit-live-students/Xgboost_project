# Let's Continue with Extreme Gradient Boosting  Parameter tuning

* Now we have tunned the first few parameter, now by using them we will tune the rest params.

## Write a function `param2` that:
* Will take following param_grid along with model, dataset that will use **myXGBoost** and will return the accuracy and best_params.

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training|
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing|
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Testing dataset target Variable |
| model | int | compulsory | | Which model needs to be build |
| param_grid | Dict | compulsory | | Dictionary of parameter |

### Return :

| Return | dtype | description |
| --- | --- | --- |
| accuracy | float | accuracy of model using those params |
| best_params | Dict | Dictionary of best fit parameter the model  |
