import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image 
#%matplotlib inline
nltk.download('stopwords')

st.title('Online Voting and Intelligence System')
st.subheader('Sentiment Analysis')
img = Image.open('/nlp_sentiment.jpg')
st.image(img)
st.markdown('Python sentiment analysis is a methodology for analyzing a piece of text to discover the sentiment hidden within it.')
st.markdown('It accomplishes this by combining machine learning and natural language processing (NLP). Sentiment analysis allows you to examine the feelings expressed in a piece of text.')
st.subheader('Electoral Debate tweets')
@st.cache
def load_data(nrows):
    debate_tweets = pd.read_csv('/Sentiment.csv',nrows=nrows)
    return debate_tweets
load_data_state = st.text('Loading data.......')
debate_tweets= load_data(1156)
load_data_state.text('Loading.........successful!')

#inspect Data
if st.checkbox('Show data'):
    st.subheader('raw data')
    st.write(debate_tweets)
st.subheader('Data preparation')
sentiment_state = st.text('Drop the unnecessary columns, keep only sentiment and text')
debate_tweets = debate_tweets[['text','sentiment']]
st.write(debate_tweets)
st.text('split the dataset into a training and a testing set. The test set is the 10% of the original dataset Drop neutral tweets since the main goal is to differentiate positive and negative ')

train,test= train_test_split(debate_tweets,test_size = 0.1)
train = train[train.sentiment !='Neutral']

st.text('separate the positive and negative tweets to visualize the words in each uding WordCloud.')
train_pos = train[train['sentiment']=='Positive']
train_pos = train_pos['text']
train_neg= train[train['sentiment']=='Negative']
train_neg = train['text']

def wordcloud_draw(debate_tweets,color='black'):
    words = ''.join(debate_tweets)
    clean_word = ''.join([word for word in words.split()
                            if 'http' not in word and not word.startswith('@')
                            and not word.startswith('#') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color=color,
                            width=2500,height=2000).generate(clean_word)
    #plt.figure(1,figsize=(13,13))
    plt.imshow(wordcloud)
    plt.axis('off')
    #plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    

st.write('Positive words')
wordcloud_draw(train_pos,'white')
st.write('Negative words')
wordcloud_draw(train_neg)

st.markdown('Stop Words are words which do not contain important significance to be used in Search Queries. Usually these words are filtered out from search queries because they return vast amount of unnecessary information. ( the, for, this etc. )')
tweets =[]
stopwords_set = set(stopwords.words('english'))
for index, row in train.iterrows():
    word_filter = [e.lower() for e in row.text.split() if len(e) >= 3]
    word_clean = [word for word in word_filter if 'http' not in word and not word.startswith('@') and not word.startswith('#') and word != 'RT']
    words_without_stopwords = [word for word in word_clean if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg = test_neg['text']

#As a next step I extracted the so called features with nltk lib, first by measuring a frequent distribution and by selecting the resulting keys.
## Extracting word features
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s' %word] = (word in document_words)
    return features
wordcloud_draw(w_features)
# Training the Naive Bayes classifier
training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
st.write(classifier)

#In this project I was curious how well nltk and the NaiveBayes Machine Learning algorithm performs for Sentiment Analysis. In my experience, it works rather well for negative comments. The problems arise when the tweets are ironic, sarcastic has reference or own difficult context.
#Consider the following tweet: "Muhaha, how sad that the Liberals couldn't destroy Trump. Marching forward." As you may already thought, the words sad and destroy highly influences the evaluation, although this tweet should be positive when observing its meaning and context.