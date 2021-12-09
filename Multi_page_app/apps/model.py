import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#cleanTxt function

def cleanTxt(text):
    text=re.sub(r'@[A-Za-z0-9]+','',text) ## removing @ mention
    text=re.sub(r'#','',text)             ## removing # symbol
    text=re.sub(r':','',text)             ## removing : symbol
    text=re.sub(r'RT[\s]+','',text)  ## removing RT followed byspace
 
  #df=df[~df.Tweets.str.contains('RT')] --> another way to remove RT
    text=re.sub(r'https?:\/\/\S+','',text) ## removing https
    return text

def app():
    st.title('Model')

    st.write('This is the `Model` page of the multi-page app.')

    st.write('The model performance of the Tweeter dataset is presented below.')
    # Load iris dataset
    df = pd.read_csv("https://raw.githubusercontent.com/MfundoMhlanga/classification-predict-streamlit-template/master/train.csv")
    df_test = pd.read_csv("https://raw.githubusercontent.com/MfundoMhlanga/classification-predict-streamlit-template/master/test_with_no_labels.csv")
    # Cleaning the messages

    df['message']=df['message'].apply(cleanTxt) 

    X = df['message']
    Y = df.sentiment

    # Model building
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #tokenization
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.95)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
    tfidf_test = tfidf_vectorizer.transform(X_test.values)

    classifier_name = st.sidebar.selectbox('Select classifier',('KNN', 'SVM', 'Random Forest'))
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        clf = SVC(C=params['C'])
        clf.fit(tfidf_train, y_train)
        st.header('Model performance')
        st.markdown('**2.1. Training set**')
        y_pred_train = clf.predict(tfidf_train)
        st.write('F1 Score:')
        st.info(f1_score(y_train, y_pred_train, average='macro'))

        st.markdown('**2.1. Training set**')
        y_pred_test = clf.predict(tfidf_test)
        st.write('F1 Score:')
        st.info(f1_score(y_test, y_pred_test, average='macro'))
        
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
        clf = KNeighborsClassifier(n_neighbors=params['K'])
        clf.fit(tfidf_train, y_train)
        st.header('Model performance')
        st.markdown('**2.1. Training set**')
        y_pred_train = clf.predict(tfidf_train)
        st.write('F1 Score:')
        st.info(f1_score(y_train, y_pred_train, average='macro'))

        st.markdown('**2.1. Training set**')
        y_pred_test = clf.predict(tfidf_test)
        st.write('F1 Score:')
        st.info(f1_score(y_test, y_pred_test, average='macro'))
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
        max_depth=params['max_depth'], random_state=1234)
        clf.fit(tfidf_train, y_train)
        st.header('Model performance')
        st.markdown('**2.1. Training set**')
        y_pred_train = clf.predict(tfidf_train)
        st.write('F1 Score:')
        st.info(f1_score(y_train, y_pred_train, average='macro'))

        st.markdown('**2.1. Training set**')
        y_pred_test = clf.predict(tfidf_test)
        st.write('F1 Score:')
        st.info(f1_score(y_test, y_pred_test, average='macro'))

    