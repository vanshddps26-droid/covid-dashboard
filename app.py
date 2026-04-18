import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv('C:/Users/Hp/Downloads/covid-19/covid_19_india.csv')
df['Date'] = pd.to_datetime(df['Date'])

st.title("COVID-19 Data Dashboard (India)")

# Total summary
latest = df[df['Date'] == df['Date'].max()]

total_confirmed = latest['Confirmed'].sum()
total_deaths = latest['Deaths'].sum()
total_cured = latest['Cured'].sum()

st.subheader("Latest Summary")
st.write("Total Confirmed:", total_confirmed)
st.write("Total Deaths:", total_deaths)
st.write("Total Recovered:", total_cured)

# Total cases graph
total_cases = df.groupby('Date')['Confirmed'].sum()
st.subheader("Total Cases Over Time")
st.line_chart(total_cases)

# State selection
state = st.selectbox("Select State", df['State/UnionTerritory'].unique())

state_data = df[df['State/UnionTerritory'] == state]
state_trend = state_data.groupby('Date')['Confirmed'].sum()

st.subheader(f"COVID Trend in {state}")
st.line_chart(state_trend)