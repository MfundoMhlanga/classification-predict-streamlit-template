import streamlit as st
import numpy as np
import pandas as pd

def app():
    st.title('Data')

    st.write("This is the `Data` page of the multi-page app.")

    st.write("The following is the DataFrame of the `tweeter` dataset.")

    # Load iris dataset
    df = pd.read_csv("https://raw.githubusercontent.com/MfundoMhlanga/classification-predict-streamlit-template/master/train.csv")
    df_test = pd.read_csv("https://raw.githubusercontent.com/MfundoMhlanga/classification-predict-streamlit-template/master/test_with_no_labels.csv")

    st.write(df.head(15))