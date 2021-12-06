import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, preprocessing

class Modeler:
    def __init__(self):
        pass

    def split_train_test(self, df, train_end_date):
        """
        Takes the df and the end date of the training range and returns 
        training dataset and testing dataset. Drops the 'date' column.
        Expects the 2 target columsn to be called 'cases' and 'deaths'
        Expects to have a 'date' column that is of time datetime
        Returns:
        X_train, y_train_cases, y_train_deaths, X_test, y_test_cases, y_test_deaths
        """
        train_mask = (df['date'] <= train_end_date)
        test_mask = ~train_mask

        X_train = df[train_mask].drop(['date', 'cases', 'deaths'], axis=1)
        y_train_cases =  df[train_mask]['cases']
        y_train_deaths = df[train_mask]['deaths']

        X_test = df[test_mask].drop(['date', 'cases', 'deaths'], axis=1)
        y_test_cases =  df[test_mask]['cases']
        y_test_deaths = df[test_mask]['deaths']

        return X_train, y_train_cases, y_train_deaths, X_test, y_test_cases, y_test_deaths


    def cv_model(self, model, X, y):
        """
        Performs 5-fold cross validation on model and reports the mean and std dev of
        'neg_root_mean_squared_error', 'r2', and 'explained_variance' for the 5 folds
        as a dictionary
        """

        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=1)

        nRMSE, r2, var = [], [], []

        for train_index, test_index in kfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            nRMSE += [metrics.mean_squared_error(y_test, y_pred, squared=False)]
            r2 += [metrics.r2_score(y_test, y_pred)]
            var += [metrics.explained_variance_score(y_test, y_pred)]

        scores_dict = {
            'RMSE' : (np.mean(nRMSE), np.std(nRMSE)) # Tuple of neg MSE (mean, std) across 5 folds
            , 'r2': (np.mean(r2), np.std(r2)) # Tuple of r2 (mean, std) across 5 folds
            , 'Explained Variance': (np.mean(var), np.std(var)) # Tuple of r2 (mean, std) across 5 folds    
        }
        return scores_dict

    def test_model(self, model, X_train, y_train, X_test, y_test):
        """
        Trains the model using the entire training dataset and tests on the test dataset
        Returns:
        y_pred, rmse, r2
        """

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        r2 = metrics.r2_score(y_test, y_pred)
        print('Model testing scores:')
        print('RMSE: %.2f' % rmse)
        print('R^2 Score: %.2f' % r2)

        return y_pred, rmse, r2




