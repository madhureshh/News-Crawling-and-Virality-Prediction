#Crawling and downloading news
import requests
from bs4 import BeautifulSoup
from newspaper import Article  
import csv 
import pandas as pd
import numpy as np

url = "https://timesofindia.indiatimes.com/world"
r = requests.get(url)

soup = BeautifulSoup(r.content, 'html5lib') 
table = soup.findAll('a', attrs = {'class':'w_img'})


news=[]
for row in table: 
    if not row['href'].startswith('http'):
        news.append('https://timesofindia.indiatimes.com'+row['href'])


df=[]
for i in news:
    article = Article(i, language="en")
    article.download() 
    article.parse() 
    article.nlp() 
    data={}
    data['Title']=article.title
    data['Text']=article.text
    data['Summary']=article.summary
    data['Keywords']=article.keywords
    df.append(data)

dataset=pd.DataFrame(df)
print(dataset.head())

#Model Building usin Random Forest
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


#converting dataset into required format
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))


def rate_unique(words):
    words=tokenize(words)
    no_order = list(set(words))
    rate_unique=len(no_order)/len(words)
    return rate_unique

def rate_nonstop(words):
    words=tokenize(words)
    filtered_sentence = [w for w in words if not w in stopwords]
    rate_nonstop=len(filtered_sentence)/len(words)
    no_order = list(set(filtered_sentence))
    rate_unique_nonstop=len(no_order)/len(words)
    return rate_nonstop,rate_unique_nonstop

def avg_token(words):
    words=tokenize(words)
    length=[]
    for i in words:
        length.append(len(i))
    return np.average(length)

from textblob import TextBlob

import datefinder
import datetime  
from datetime import date 
def day(article_text):
    article=article_text
    if len(list(datefinder.find_dates(article)))>0:
        date=str(list(datefinder.find_dates(article))[0])
        date=date.split()
        date=date[0]
        year, month, day = date.split('-')     
        day_name = datetime.date(int(year), int(month), int(day)) 
        return day_name.strftime("%A")
    return "Monday"

def tokenize(text):
    text=text
    return word_tokenize(text)

pos_words=[]
neg_words=[]
def polar(words):
    all_tokens=tokenize(words)
    for i in all_tokens:
        analysis=TextBlob(i)
        polarity=analysis.sentiment.polarity
        if polarity>0:
            pos_words.append(i)
        if polarity<0:
            neg_words.append(i)
    return pos_words,neg_words


def rates(words):
    words=polar(words)
    pos=words[0]
    neg=words[1]
    all_words=words
    global_rate_positive_words=(len(pos)/len(all_words))/100
    global_rate_negative_words=(len(neg)/len(all_words))/100
    pol_pos=[]
    pol_neg=[]
    for i in pos:
        analysis=TextBlob(i)
        pol_pos.append(analysis.sentiment.polarity)
        avg_positive_polarity=analysis.sentiment.polarity
    for j in neg:
        analysis2=TextBlob(j)
        pol_neg.append(analysis2.sentiment.polarity)
        avg_negative_polarity=analysis2.sentiment.polarity
    min_positive_polarity=min(pol_pos)
    max_positive_polarity=max(pol_pos)
    min_negative_polarity=min(pol_neg)
    max_negative_polarity=max(pol_neg)
    avg_positive_polarity=np.average(pol_pos)
    avg_negative_polarity=np.average(pol_neg)
    return global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,min_positive_polarity,max_positive_polarity,avg_negative_polarity,min_negative_polarity,max_negative_polarity


df2=[]
for i in news:
    pred_info={}
    article = Article(i, language="en") # en for English 
    article.download() 
    article.parse()
    analysis=TextBlob(article.text)
    polarity=analysis.sentiment.polarity
    title_analysis=TextBlob(article.title)
    pred_info['text']=article.text
    pred_info['n_tokens_title']=len(tokenize(article.title))
    pred_info['n_tokens_content']=len(tokenize(article.text))
    pred_info['n_unique_tokens']=rate_unique(article.text)
    pred_info['n_non_stop_words']=rate_nonstop(article.text)[0]
    pred_info['n_non_stop_unique_tokens']=rate_nonstop(article.text)[1]
    pred_info['num_hrefs']=article.html.count("https://timesofindia.indiatimes.com")
    pred_info['num_imgs']=len(article.images)
    pred_info['num_videos']=len(article.movies)
    pred_info['average_token_length']=avg_token(article.text)
    pred_info['num_keywords']=len(article.keywords)
    
    if "life-style" in article.url:
        pred_info['data_channel_is_lifestyle']=1
    else:
        pred_info['data_channel_is_lifestyle']=0
    if "etimes" in article.url:
        pred_info['data_channel_is_entertainment']=1
    else:
        pred_info['data_channel_is_entertainment']=0
    if "business" in article.url:
        pred_info['data_channel_is_bus']=1
    else:
        pred_info['data_channel_is_bus']=0
    if "social media" or "facebook" or "whatsapp" in article.text.lower():
        data_channel_is_socmed=1
        data_channel_is_tech=0
        data_channel_is_world=0
    else:
        data_channel_is_socmed=0
    if ("technology" or "tech" in article.text.lower()) or ("technology" or "tech" in article.url):
        data_channel_is_tech=1
        data_channel_is_socmed=0
        data_channel_is_world=0
    else:
        data_channel_is_tech=0
    if "world" in article.url:
        data_channel_is_world=1
        data_channel_is_tech=0
        data_channel_is_socmed=0
    else:
        data_channel_is_world=0
        
    pred_info['data_channel_is_socmed']=data_channel_is_socmed
    pred_info['data_channel_is_tech']=data_channel_is_tech
    pred_info['data_channel_is_world']=data_channel_is_world
    
    if day(i)=="Monday":
        pred_info['weekday_is_monday']=1
    else:
        pred_info['weekday_is_monday']=0
    if day(i)=="Tuesday":
        pred_info['weekday_is_tuesday']=1
    else:
        pred_info['weekday_is_tuesday']=0
    if day(i)=="Wednesday":
        pred_info['weekday_is_wednesday']=1
    else:
        pred_info['weekday_is_wednesday']=0
    if day(i)=="Thursday":
        pred_info['weekday_is_thursday']=1
    else:
        pred_info['weekday_is_thursday']=0
    if day(i)=="Friday":
        pred_info['weekday_is_friday']=1
    else:
        pred_info['weekday_is_friday']=0
    if day(i)=="Saturday":
        pred_info['weekday_is_saturday']=1
        pred_info['is_weekend']=1
    else:
        pred_info['weekday_is_saturday']=0
    if day(i)=="Sunday":
        pred_info['weekday_is_sunday']=1
        pred_info['is_weekend']=1
    else:
        pred_info['weekday_is_sunday']=0
        pred_info['is_weekend']=0
        
    pred_info['global_subjectivity']=analysis.sentiment.subjectivity
    pred_info['global_sentiment_polarity']=analysis.sentiment.polarity
    pred_info['global_rate_positive_words']=rates(article.text)[0]
    pred_info['global_rate_negative_words']=rates(article.text)[1]
    pred_info['avg_positive_polarity']=rates(article.text)[2]
    pred_info['min_positive_polarity']=rates(article.text)[3]
    pred_info['max_positive_polarity']=rates(article.text)[4]
    pred_info['avg_negative_polarity']=rates(article.text)[5]
    pred_info['min_negative_polarity']=rates(article.text)[6]
    pred_info['max_negative_polarity']=rates(article.text)[7]    
    pred_info['title_subjectivity']=title_analysis.sentiment.subjectivity
    pred_info['title_sentiment_polarity']=title_analysis.sentiment.polarity
    df2.append(pred_info)

pred_df=pd.DataFrame(df2)
pred_test=pred_df.drop(['text'],axis=1)
pred_df.head()

#predicting final virality
test2=pd.DataFrame(clf.predict(pred_test),pred_df['text'])
test2.reset_index(level=0, inplace=True)
test2 = test2.rename(index=str, columns={"index": "News", 0: "Virality"})
test2
