import streamlit as st
import datetime
from datetime import date

st.title("Stock Prediction App")
st.subheader('This app is created to forecast the stock market movement using daily news headlines')
st.image("https://t4.ftcdn.net/jpg/02/20/32/75/240_F_220327557_gRDTuYL4iVG0lWrjgjrv1chBCUunjKlG.jpg")

news_headline = st.text_input("Enter News Headline")

stockprice_yesterday = st.text_input("Closing Stock Price From Yesterday")

data_load_state = st.text("Load data...")

