import csv 
import pandas as pd
import numpy as np


def clean_cols(data):
    """Clean the column names by stripping and lowercase."""
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)
'''
def TrainTestSplit(X, Y, R=0, test_size=0.2):
    """Easy Train Test Split call."""
    return train_test_split(X, Y, test_size=test_size, random_state=R)
'''
full_data = clean_cols(pd.read_csv("OnlineNewsPopularity.csv"))
X = full_data.drop(['url','shares'], axis = 1)
y = full_data['shares']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 300,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
rf.fit(X_train, y_train)

# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df
X_train = pd.DataFrame(X_train)
base_imp = imp_df(X_train.columns, rf.feature_importances_)
base_imp
    
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
            .set_title(title, fontsize = 20)
var_imp_plot(base_imp, 'Default feature importance (scikit-learn)')
base_imp = pd.DataFrame(X)
base_imp.to_excel (r'crawl1.xlsx', index = False, header=True)


