import pandas as pd
import numpy as np
import pathlib
import re
import matplotlib.pyplot as plt
from transformers import BertTokenizer

datapath = pathlib.Path(__file__).parent.parent

dataframes = []
print("Concatenating Trolls")
for i in range(9):
    dataframes.append(pd.read_csv(datapath/f"data/trolls/IRAhandle_tweets_{i+1}.csv"))
trolldata = pd.concat(dataframes)

print("Reading Genuine")
genuine = pd.read_csv(datapath/"data/genuine/Political_tweets.csv")

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
print("Normalizing")
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
def count_capitals(text):
    num_capitals = len(re.findall(r'[A-Z]', text))
    return num_capitals
def count_numerics(text):
    num_numerics = len(re.findall(r'\d', text))
    return num_numerics

def count_special_characters(text):
    num_special_chars = len(re.findall(r'[!?$%^&*]', text))
    return num_special_chars

def count_exclamations(text):
    num_exclamations = len(re.findall(r'!', text))
    return num_exclamations

def average_word_length(text):
    words = text.split()
    if not words: 
        return 0
    return sum(len(word) for word in words) / len(words)

def count_sentences(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    return num_sentences

print("Counting Hashtags")
bigdataset['hash'] = bigdataset['content'].apply(counthash)
print("Counting Links")
bigdataset['link'] = bigdataset['content'].apply(countlinks)
print("Counting Capitals")
bigdataset['capitals'] = bigdataset['content'].apply(count_capitals)
print("Counting Numerics")
bigdataset['numerics'] = bigdataset['content'].apply(count_numerics)
print("Counting Special Characters")
bigdataset['special_chars'] = bigdataset['content'].apply(count_special_characters)
print("Counting Exclamations")
bigdataset['exclamations'] = bigdataset['content'].apply(count_exclamations)
print("Calculating Average Word Length")
bigdataset['avg_word_length'] = bigdataset['content'].apply(average_word_length)
print("Counting Sentences")
bigdataset['sentences'] = bigdataset['content'].apply(count_sentences)


# bigdataset[]
bigdataset.reset_index(inplace=True,drop=True)

bigdataset.to_csv(datapath/"data/transformed/data.csv")


bigdataset = pd.read_csv(datapath/"data/transformed/data.csv")
bigdataset.dropna(inplace=True)
bigdataset.to_csv(datapath/"data/transformed/data.csv")

