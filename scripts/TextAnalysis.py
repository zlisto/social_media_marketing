import re
import numpy as np
import pandas as pd


#clean tweet text for doing TF, TF-IDF embeddings
def clean_tweet(elem):
    s1 =re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?|u2026|u2019|u2019s|u2014|amp", "", elem.lower())
    return s1.strip()

def engagement_estimate(df):
    data = df[['like_count','retweet_count','reply_count','impression_count','keyword']].groupby(by = 'keyword').sum()

    nlike = data.like_count.sum()
    nretweet = data.retweet_count.sum()
    nreply = data.reply_count.sum()
    
    vc = data.impression_count
    nc = data.like_count + data.retweet_count + data.reply_count
    n = nlike + nretweet + nreply

    alphac = nlike/n * nc/vc
    df_alpha = alphac.to_frame(name='alpha').reset_index(drop=False)
    
  
    beta_like = nlike/nlike
    beta_retweet = nretweet/nlike
    beta_reply = nreply/nlike
    
    df_beta = pd.DataFrame({'interaction':['like','retweet','reply'],
                             'beta':[beta_like, beta_retweet, beta_reply]})
    return df_alpha, df_beta