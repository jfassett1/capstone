import pandas as pd
import numpy as np

import re
import matplotlib.pyplot as plt
from transformers import BertTokenizer


dataframes = []
for i in range(9):
    dataframes.append(pd.read_csv(f"data/trolls/IRAhandle_tweets_{i+1}.csv"))
trolldata = pd.concat(dataframes)

genuine = pd.read_csv("data/genuine/Political_tweets.csv")

#Removing non-english entries
troll = trolldata[trolldata["language"] == "English"]
# genuine.iloc[:100_000]


#Getting list of troll authors
trollauthors = []
for author in troll['author'].unique():
    trollauthors.append(author)
trollauthors

#Getting list of genuine authors
genuineauthors = genuine['user_name'].apply(lambda x: x.upper() if isinstance(x, str) else x)

#Searching for identical names

matches = []
for rname,tname in zip(genuineauthors,trollauthors):
    if rname == tname:
        matches.append(rname)

print(f"Found {len(matches)} matches: {matches}")


troll.drop(columns=["language","harvested_date","publish_date","external_author_id","following","account_type","new_june_2018","region","post_type","updates","account_category"],inplace=True)
troll.drop(columns=["author"],inplace=True)
genuine.drop(columns=['user_name','user_location','user_description','user_created','user_friends','user_favourites','user_verified','date','hashtags','source'],inplace=True)

namechanges = {
"user_followers":"followers",
"is_retweet":"retweet",
"text":"content",

}

genuine.rename(columns=namechanges,inplace=True)

genlabels = np.zeros(shape=(len(genuine),1))


trolabels = np.ones(shape=(len(troll),1))

genuine['troll'] = genlabels
troll['troll'] = trolabels
genuine['retweet'] = genuine['retweet'].apply(lambda x: 0 if x == False else 1)

bigdataset = pd.concat([genuine.reset_index(drop=True), troll.reset_index(drop=True)], axis=0)

x = bigdataset['followers']
def logscaling(x):
    scaled = np.log(x+1)
    return scaled

logscaled = logscaling(x).value_counts()

# plt.bar(logscaled.index,height=logscaled)
# plt.xlabel("Followers")
# plt.ylabel("Users")
# plt.title("Distribution of Followers (Log scaled)")
# plt.show()


#Normalizing Dataset
bigdataset['followers'] = logscaling(bigdataset['followers'])



#Removes any instance of a non-string input. I decided this was better than just stringifying inputs because numerical inputs aren't useful.
bigdataset['content'] = bigdataset['content'].apply(lambda x: np.nan if type(x) != str else x )
bigdataset.dropna(inplace=True)

def preprocess_text(text):
    # Remove newline characters
    
    text = re.sub(r'\n+', ' ', text)
    # Replacing URLs
    text = re.sub(r'https?://\S+', '', text)
    # Replacing Hashtags
    text = re.sub(r'#\w+', '', text)
    return text.strip()

def counthash(text):
    num_hashtags = len(re.findall(r'#\w+', text))
    return num_hashtags
def countlinks(text):
    num_links = len(re.findall(r'https?://\S+', text))
    return num_links
#Applying function
bigdataset['hash'] = bigdataset['content'].apply(counthash)
bigdataset['link'] = bigdataset['content'].apply(counthash)
bigdataset['content'] = bigdataset['content'].apply(preprocess_text)

# bigdataset[]
bigdataset.reset_index(inplace=True,drop=True)

bigdataset.to_csv("data/transformed/data.csv")


bigdataset = pd.read_csv("data/transformed/data.csv")
bigdataset.dropna(inplace=True)
bigdataset.to_csv("data/transformed/data.csv")

