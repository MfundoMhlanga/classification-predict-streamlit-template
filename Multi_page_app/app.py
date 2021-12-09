import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, User_Input # import your app modules here

app = MultiApp()

st.markdown("""
# The Tweeter Classifier App
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
app.add_app("User Input", User_Input.app)
# The main app
app.run() 