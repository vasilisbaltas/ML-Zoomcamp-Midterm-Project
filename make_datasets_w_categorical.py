#  The scope of this notebook is to introduce a utility function that would be really helpful when dealing with time series data.
#  The function transforms the series and converts them to a tabular form (tabularize) so that they are suitable for machine learning
#  development. This means that we can feed a classic ML regressor with a matrix of X features and a matrix of Y target variables.
#  In this way we preserve the typical F(X) ~ Y problem formulation. The function also provides us with training and test sets so that we can train
#  vour models or produce future forecasts (after the end of the available data)
#  Moreover, the function can  handle and add categorical exogenous variables
#  which is extremely helpful since we usually want to enhance our predictions by incorporating more variables(i.e. the car brand, the country of sales,        #  distribution channel...)

import pandas as pd
import numpy as np



def make_datasets_w_categorical(df, x_len, y_len, test_loops=12, cat_names = [] ):
    """
    Performs train test split for tabularized time series

    :param df:             dataframe where each row corresponds to a brand/product/etc. with its actual values in the vertical axis(columns)
    :param x_len:          number of historical periods we will use to make predictions (time window)
    :param y_len:          number of future periods we will predict (forecasting horizon)
    :param test_loops:     the number of loops/iterations we want to leave out(for h-tuning, validation ...).Set to 0 in order to populate future forecast
    :param cat_names:      list that contains the names of categorical values we want to include in the forecasting process
    :return:               numpy arrays, x_train, y_train, x_test, y_test
    """

    col_categorical = [col for col in df.columns if col in cat_names]
    S = df.drop(columns = col_categorical).values
    C = df[col_categorical].values                    # caterorical values

    rows, periods = S.shape

    # create train set
    loops = periods + 1 -x_len - y_len

    train = []
    for col in range(loops):
       train.append(S[:, col:col+x_len+y_len])
    train = np.vstack(train)

    X_train, y_train = np.split(train, [-y_len], axis=1)
    X_train = np.hstack((np.vstack([C]*loops), X_train))      # horizontally concatenate the categorical values


    # create test set
    if test_loops > 0 :
       X_train, X_test = np.split(X_train, [-rows*test_loops], axis=0)
       y_train, y_test = np.split(y_train, [-rows*test_loops], axis=0)
    else:
       X_test = np.hstack((C, S[:, -x_len:]))
       y_test = np.full((X_test.shape[0], y_len), np.nan)      # dummy values

    if y_len == 1:
        y_train = y_train.ravel()
        y_test  = y_test.ravel()

    return X_train, y_train, X_test, y_test