import streamlit as st
import datetime
from datetime import date

st.title("Stock Prediction App")
st.subheader('This app is created to forecast the stock market movement using daily news headlines')
st.image("https://t4.ftcdn.net/jpg/02/20/32/75/240_F_220327557_gRDTuYL4iVG0lWrjgjrv1chBCUunjKlG.jpg")

st.session_state.News = ""
st.session_state.Price = ""

news_headline = st.text_input("Enter News Headline")
stockprice_yesterday = st.text_input("Yesterdays closing Stock Price")

st.session_state.News = news_headline
st.session_state.Price = stockprice_yesterday

data_load_state = st.text("Load data...")

