import csv 
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def clean_cols(data):
    """Clean the column names by stripping and lowercase."""
    clean_col_map = {x: x.lower().strip() for x in list(data)}
    return data.rename(index=str, columns=clean_col_map)

def TrainTestSplit(X, Y, R=0, test_size=0.2):
    """Easy Train Test Split call."""
    return train_test_split(X, Y, test_size=test_size, random_state=R)


full_data = clean_cols(pd.read_csv("OnlineNewsPopularity.csv"))
train_set, test_set = train_test_split(full_data, test_size=0.20, random_state=42)

x_train = train_set.drop(['url','shares', 'timedelta', 'lda_00','lda_01','lda_02','lda_03','lda_04','num_self_hrefs', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','rate_positive_words','rate_negative_words','abs_title_subjectivity','abs_title_sentiment_polarity'], axis=1)
y_train = train_set['shares']

x_test = test_set.drop(['url','shares', 'timedelta', 'num_self_hrefs', 'kw_min_min', 'kw_max_min', 'kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','rate_positive_words','rate_negative_words','abs_title_subjectivity','abs_title_sentiment_polarity'], axis=1)
y_test = test_set['shares']

clf = RandomForestRegressor(random_state=42)
clf.fit(x_train, y_train)

rf_res = pd.DataFrame(clf.predict(x_train),list(y_train))


rf_res.reset_index(level=0, inplace=True)
rf_res_df = rf_res.rename(index=str, columns={"index": "Actual shares", 0: "Predicted shares"})
rf_res_df.head()
