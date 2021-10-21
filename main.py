from string import punctuation
import spacy
from newsapi import NewsApiClient
import pandas as pd
import string
from collections import Counter
import pickle

filename = 'articlesCOVID.pckl'
pos_tag = ('VERB', 'NOUN', 'PROPN')
result = []
def get_keywords_eng(doc):
    for token in doc:
        if(token in nlp_eng.Defaults.stop_words or token.text in string.punctuation or token.text == 'chars'):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)

def load_files():
    return pickle.load(open(filename, 'rb'))

nlp_eng = spacy.load('en_core_web_lg')
newsapi = NewsApiClient(api_key='371f18a638d14bdd8267b996d1f58fa9')

temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-09-22', to='2021-10-22', sort_by='relevancy', page_size=100)
pickle.dump(temp['articles'], open(filename, 'wb'))

article_data = []
for i, article in enumerate(temp['articles']):
    title = article['title']
    description = article['description']
    content = article['content']
    article_data.append({'title':title, 'description':description, 'content':content})

df = pd.DataFrame(article_data)
pd.set_option('display.max_rows', df.shape[0]+1)
df = df.dropna()
df.head()

for article in article_data:
    doc = nlp_eng(article['content'])
    get_keywords_eng(doc)
print(Counter(result).most_common(5))

