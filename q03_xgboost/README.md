# Build a Xgboost using the bestmodel

* Now we have tunned the parameter, we gonna use them in the model.

## Write a function `xgboost` that:
* Will take following dataset and will return the accuracy and best_params.


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training|
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing|
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Testing dataset target Variable |
| **kwargs |  | compulsory | | additional parameter to be given |

### Return :

| Return | dtype | description |
| --- | --- | --- |
| accuracy | float | accuracy of model using those params |

To-Do list :

Check for different n_estimators and learning_rate and find the best score
