import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

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
    st.title('User Input')

    st.write('This is the `User Input` page of the multi-page app.')

    st.write('Text prediction.')
    text = st.text_input("Enter your text (required)")

    if not text:
        st.warning("Please fill out so required fields")

    if st.button("Classify"):

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

        clf = RandomForestClassifier()
        clf.fit(tfidf_train, y_train)
        #Clean input text 
        text = cleanTxt(text)
        text = [text]
        tfidf_text = tfidf_vectorizer.transform(text)
        y_pred_text = clf.predict(tfidf_text)
        st.write('Prediction:')
        st.info(y_pred_text)

    