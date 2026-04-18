#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/Hp/Downloads/covid-19/covid_19_india.csv')
print(df)


# In[6]:


df.replace('-', 0, inplace=True)

# Convert columns to numeric
df['ConfirmedIndianNational'] = pd.to_numeric(df['ConfirmedIndianNational'])
df['ConfirmedForeignNational'] = pd.to_numeric(df['ConfirmedForeignNational'])


# In[11]:


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)


# In[8]:


total_cases = df.groupby('Date')['Confirmed'].sum()

plt.figure()
total_cases.plot()
plt.title('Total COVID-19 Confirmed Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.show()


# In[9]:


top_states = df.groupby('State/UnionTerritory')['Confirmed'].max().sort_values(ascending=False).head(5)

plt.figure()
top_states.plot(kind='bar')
plt.title('Top 5 States with Highest COVID Cases')
plt.xlabel('State')
plt.ylabel('Cases')
plt.show()


# In[10]:


latest_data = df[df['Date'] == df['Date'].max()]

totals = latest_data[['Cured', 'Deaths']].sum()

plt.figure()
totals.plot(kind='bar')
plt.title('Total Cured vs Deaths')
plt.ylabel('Count')
plt.show()


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


latest_data = df[df['Date'] == df['Date'].max()]

total_confirmed = latest_data['Confirmed'].sum()
total_deaths = latest_data['Deaths'].sum()

death_rate = (total_deaths / total_confirmed) * 100

print("Death Rate (%):", death_rate)


# In[18]:


total_cured = latest_data['Cured'].sum()

recovery_rate = (total_cured / total_confirmed) * 100

print("Recovery Rate (%):", recovery_rate)


# In[14]:


labels = ['Cured', 'Deaths', 'Active']
active = total_confirmed - (total_cured + total_deaths)

values = [total_cured, total_deaths, active]

plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('COVID-19 Distribution')
plt.show()


# In[20]:


state = 'Maharashtra'   # change state if needed

state_data = df[df['State/UnionTerritory'] == state]
state_group = state_data.groupby('Date')['Confirmed'].sum()

plt.figure()
state_group.plot()
plt.title(f'COVID Trend in {state}')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()


# In[21]:


df['Active'] = df['Confirmed'] - df['Cured'] - df['Deaths']


# In[22]:


active_cases = df.groupby('Date')['Active'].sum()

plt.figure()
active_cases.plot()
plt.title('Active COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Active Cases')
plt.show()


# In[23]:


latest_data = df[df['Date'] == df['Date'].max()]

top_states = latest_data.sort_values(by='Confirmed', ascending=False).head(5)

plt.figure()
plt.bar(top_states['State/UnionTerritory'], top_states['Confirmed'])
plt.title('Top 5 States (Latest Data)')
plt.xlabel('State')
plt.ylabel('Cases')
plt.xticks(rotation=45)
plt.show()


# In[24]:


df = df.sort_values('Date')

# Daily new cases
df['Daily Cases'] = df.groupby('State/UnionTerritory')['Confirmed'].diff().fillna(0)

# Total daily cases (India level)
daily_cases = df.groupby('Date')['Daily Cases'].sum()

plt.figure()
daily_cases.plot()
plt.title('Daily New COVID-19 Cases in India')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()


# In[25]:


top10 = df.groupby('State/UnionTerritory')['Confirmed'].max().sort_values(ascending=False).head(10).index

plt.figure()

for state in top10:
    state_data = df[df['State/UnionTerritory'] == state]
    trend = state_data.groupby('Date')['Confirmed'].sum()
    plt.plot(trend, label=state)

plt.title('Top 10 States COVID Trend')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.show()


# In[30]:


import seaborn as sns
pivot = df.pivot_table(values='Confirmed',
                       index='State/UnionTerritory',
                       columns='Date',
                       aggfunc='max')

plt.figure(figsize=(12,8))
sns.heatmap(pivot.fillna(0), cmap='viridis')
plt.title('State-wise COVID Heatmap')
plt.show()


# In[27]:


df['Month'] = df['Date'].dt.to_period('M')

monthly = df.groupby('Month')['Confirmed'].max()

plt.figure()
monthly.plot(kind='bar')
plt.title('Monthly COVID Growth')
plt.xlabel('Month')
plt.ylabel('Cases')
plt.show()


# In[29]:


corr = df[['Confirmed', 'Cured', 'Deaths']].corr()

import seaborn as sns
plt.figure()
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cases = pd.read_csv('C:/Users/Hp/Downloads/covid-19/covid_19_india.csv')
print(cases)


# In[4]:


cases.replace('-', 0, inplace=True)

cases['Date'] = pd.to_datetime(cases['Date'], dayfirst=True)

cases['Active'] = cases['Confirmed'] - cases['Cured'] - cases['Deaths']
print(cases.head())


# In[5]:


india = cases.groupby('Date')['Confirmed'].sum()

# 7-day moving average
rolling_avg = india.rolling(window=7).mean()

plt.figure()
india.plot(label='Daily Cases')
rolling_avg.plot(label='7-Day Average')
plt.legend()
plt.title('COVID Trend with Moving Average')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()


# In[6]:


print("Highest Cases:", india.max())
print("Peak Date:", india.idxmax())


# In[7]:


peak_date = india.idxmax()

peak_data = cases[cases['Date'] == peak_date]

top_state = peak_data.sort_values(by='Confirmed', ascending=False).head(5)

print(top_state[['State/UnionTerritory', 'Confirmed']])


# In[13]:


from sklearn.linear_model import LinearRegression
import numpy as np

# India total cases
india = cases.groupby('Date')['Confirmed'].sum().reset_index()

# Convert date to numeric
india['Day'] = np.arange(len(india))

# Train model
model = LinearRegression()
model.fit(india[['Day']], india['Confirmed'])

# Predict next 10 days
future_days = pd.DataFrame(np.arange(len(india), len(india)+10), columns=['Day'])

predictions = model.predict(future_days)

print("Future Predictions:", predictions)


# In[9]:


plt.figure()

plt.plot(india['Confirmed'], label='Actual Data')
plt.plot(range(len(india), len(india)+10), predictions, label='Predicted Data')

plt.legend()
plt.title('COVID Future Prediction')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()


# In[14]:


plt.figure()

plt.plot(india['Day'], india['Confirmed'], label='Actual Cases')
plt.plot(future_days['Day'], predictions, linestyle='--', label='Predicted Cases')

plt.legend()
plt.title('COVID-19 Future Prediction (India)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.show()


# In[15]:


india_data = cases.groupby('Date')[['Confirmed', 'Cured', 'Deaths', 'Active']].sum()

plt.figure()
india_data.plot()
plt.title('COVID-19 Overall Comparison')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()


# In[16]:


india_data['Death Rate'] = (india_data['Deaths'] / india_data['Confirmed']) * 100

plt.figure()
india_data['Death Rate'].plot()
plt.title('Death Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.show()


# In[17]:


india_data['Recovery Rate'] = (india_data['Cured'] / india_data['Confirmed']) * 100

plt.figure()
india_data['Recovery Rate'].plot()
plt.title('Recovery Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.show()


# In[18]:


import seaborn as sns

top10 = cases.groupby('State/UnionTerritory')['Confirmed'].max().sort_values(ascending=False).head(10).index

filtered = cases[cases['State/UnionTerritory'].isin(top10)]

pivot = filtered.pivot_table(values='Confirmed',
                             index='State/UnionTerritory',
                             columns='Date',
                             aggfunc='max')

plt.figure(figsize=(12,6))
sns.heatmap(pivot.fillna(0))
plt.title('Top 10 States Heatmap')
plt.show()


# In[19]:


cases['Daily Cases'] = cases.groupby('State/UnionTerritory')['Confirmed'].diff().fillna(0)

india_daily = cases.groupby('Date')['Daily Cases'].sum()

plt.figure()
india_daily.plot()
plt.title('COVID Waves in India')
plt.xlabel('Date')
plt.ylabel('Daily Cases')
plt.show()


# In[20]:


latest = cases[cases['Date'] == cases['Date'].max()]

top5 = latest.sort_values(by='Confirmed', ascending=False).head(5)

labels = top5['State/UnionTerritory']
values = top5['Confirmed']

plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Top 5 States Contribution')
plt.show()


# In[21]:


print("Total Confirmed:", india_data['Confirmed'].max())
print("Total Deaths:", india_data['Deaths'].max())
print("Total Recovered:", india_data['Cured'].max())

print("Highest Death Rate:", india_data['Death Rate'].max())
print("Highest Recovery Rate:", india_data['Recovery Rate'].max())


# In[22]:


threshold = india_daily.mean() + 2 * india_daily.std()

spikes = india_daily[india_daily > threshold]

print("Spike Dates:\n", spikes)


# In[23]:


peaks = india_daily.sort_values(ascending=False).head(5)

print("Top Peak Days:\n", peaks)


# In[24]:


plt.figure()

india_daily.plot(label='Daily Cases')

# Mark peaks
for date in peaks.index:
    plt.axvline(x=date, linestyle='--')

plt.title('COVID Waves Detection in India')
plt.xlabel('Date')
plt.ylabel('Daily Cases')
plt.legend()
plt.show()


# In[25]:


india_growth = india_daily.pct_change() * 100

plt.figure()
india_growth.plot()
plt.title('Daily Growth Rate (%)')
plt.xlabel('Date')
plt.ylabel('Growth %')
plt.show()


# In[26]:


print("Worst Day:", india_daily.idxmax())
print("Max Cases in a Day:", india_daily.max())


# In[27]:


latest = cases[cases['Date'] == cases['Date'].max()]

# Ranking
latest['Rank'] = latest['Confirmed'].rank(ascending=False)

# Top 10 states
top10 = latest.sort_values('Rank').head(10)

print(top10[['State/UnionTerritory', 'Confirmed', 'Rank']])


# In[28]:


plt.figure()

plt.barh(top10['State/UnionTerritory'], top10['Confirmed'])
plt.title('Top 10 States by COVID Cases')
plt.xlabel('Cases')
plt.ylabel('State')

plt.gca().invert_yaxis()  # Highest at top
plt.show()


# In[29]:


state_growth = cases.groupby('State/UnionTerritory')['Confirmed'].pct_change()

cases['State Growth %'] = state_growth * 100

# Show top growth states
growth_data = cases.groupby('State/UnionTerritory')['State Growth %'].mean().sort_values(ascending=False).head(10)

print(growth_data)


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prepare data
india = cases.groupby('Date')['Confirmed'].sum().reset_index()
india['Day'] = np.arange(len(india))

X = india[['Day']]
y = india['Confirmed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)


# In[32]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# In[33]:


plt.figure()

plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')

plt.legend()
plt.title('Actual vs Predicted Cases')
plt.show()


# In[ ]:




