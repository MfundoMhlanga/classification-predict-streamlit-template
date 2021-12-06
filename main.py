import numpy
import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.datasets import load_diabetes, load_boston

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Tweeter Classifier App',
    layout='wide')

#---------------------------------#

st.write("""
# The Tweeter Classifier App

Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.

To create a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data. Hence, providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories thus increasing their insights and informing future marketing strategies.

Try adjusting the hyperparameters!

""")

#Data Input 

df = pd.read_csv("https://raw.githubusercontent.com/MfundoMhlanga/classification-predict-streamlit-template/master/train.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/MfundoMhlanga/classification-predict-streamlit-template/master/test_with_no_labels.csv")

# Displays the dataset
st.subheader('1. Dataset')

st.markdown('**1.1. Glimpse of dataset**')
st.write(df)

#cleaning the data
def cleanTxt(text):
    text=re.sub(r'@[A-Za-z0-9]+','',text) ## removing @ mention
    text=re.sub(r'#','',text)             ## removing # symbol
    text=re.sub(r':','',text)             ## removing : symbol
    text=re.sub(r'RT[\s]+','',text)  ## removing RT followed byspace
 
  #df=df[~df.Tweets.str.contains('RT')] --> another way to remove RT
    text=re.sub(r'https?:\/\/\S+','',text) ## removing https
    return text
## clean Text
df['message']=df['message'].apply(cleanTxt) 
st.markdown('**1.2. Glimpse of cleaned dataset**')
st.write(df)
# Model building

## applying function
X = df.message # Using using the massage cleaned column
Y = df.sentiment # Predicting the sentiments 

#---------------------------------#

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

# Data splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

#vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.95)
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
tfidf_test = tfidf_vectorizer.transform(X_test.values)
#tfidf_df_test = tfidf_vectorizer.transform(df_test['clean'].values)

rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
    random_state=parameter_random_state,
    max_features=parameter_max_features,
    criterion=parameter_criterion,
    min_samples_split=parameter_min_samples_split,
    min_samples_leaf=parameter_min_samples_leaf,
    bootstrap=parameter_bootstrap,
    oob_score=parameter_oob_score,
    n_jobs=parameter_n_jobs)
rf.fit(tfidf_train, Y_train)

st.subheader('2. Model Performance')

st.markdown('**2.1. Training set**')
Y_pred_train = rf.predict(tfidf_train)
st.write('Coefficient of determination ($R^2$):')
st.info( r2_score(Y_train, Y_pred_train) )

st.write('Error (MSE or MAE):')
st.info( mean_squared_error(Y_train, Y_pred_train) )

st.write('F1 Score (MSE or MAE):')
st.info( f1_score(Y_test, Y_pred_train, average= 'macro') )

st.markdown('**2.2. Test set**')
Y_pred_test = rf.predict(tfidf_test)
st.write('Coefficient of determination ($R^2$):')
st.info( r2_score(Y_test, Y_pred_test) )

st.write('Error (MSE or MAE):')
st.info( mean_squared_error(Y_test, Y_pred_test) )

st.write('F1 Score (MSE or MAE):')
st.info( f1_score(Y_test, Y_pred_test, average= 'macro') )

