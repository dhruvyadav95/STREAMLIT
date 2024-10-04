import streamlit as st
import yfinance as yf
import datetime 

ticker_symbol=st.text_input("enter stock name","AAPL")
start_date = st.date_input("START DATE",value= datetime.date(2019, 7, 6))
end_date= st.date_input("END DATE", value=datetime.date(2023, 7, 6))
data = yf.download(ticker_symbol,start=start_date,end=end_date)
st.write(data)

st.line_chart(data["Close"])
st.bar_chart(data["Close"])