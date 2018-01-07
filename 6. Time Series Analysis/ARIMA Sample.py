
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm


# In[2]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


# Step 1. Prepare data.
df = pd.read_csv('monthly-milk-production-pounds-p.csv')


# In[5]:


df.columns = ['Month','Milk in Pounds per Cow']


# In[8]:


df.drop(168,axis=0,inplace=True)


# In[9]:


df['Month'] = pd.to_datetime(df['Month'])


# In[11]:


df.set_index('Month',inplace=True)


# In[13]:


df.describe().transpose()


# In[22]:


# Step 2. Plot time series data.


# In[15]:


time_series = df['Milk in Pounds per Cow']


# In[21]:


time_series.rolling(12).mean().plot(label='12 Month Rolling Mean')
time_series.rolling(12).std().plot(label='12 Month Rolling Std')
time_series.plot()
plt.legend()


# In[23]:


# Step 3. EST decomposition.
from statsmodels.tsa.seasonal import seasonal_decompose


# In[26]:


decomp = seasonal_decompose(time_series,freq=12)
fig = decomp.plot()


# In[27]:


# Step 4. Test stationarity
from statsmodels.tsa.stattools import adfuller


# In[29]:


def adf_check(time_series):
    result = adfuller(time_series)
    print("Augmented Dicky-Fuller Test")
    labels = ['ADF Test Statistic','p-value','# of lags','Num of Observations used']
    
    for value,label in zip(result,labels):
        print(label + " : " + str(value))
        
    if result[1] <= 0.05:
        print("Strong evidence against null hypothesis")
        print("Reject null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis")
        print("Failed to reject null hypothesis")
        print("Data has a unit root, it is non-stationary")


# In[30]:


adf_check(df['Milk in Pounds per Cow'])


# In[31]:


# Step 5. Differencing.


# In[32]:


df['First Difference'] = df['Milk in Pounds per Cow'] - df['Milk in Pounds per Cow'].shift(1)


# In[33]:


adf_check(df['First Difference'].dropna())


# In[34]:


df['Second Difference'] = df['First Difference'] - df['First Difference'].shift(1)


# In[35]:


adf_check(df['Second Difference'].dropna())


# In[36]:


df['Second Difference'].plot()


# In[38]:


df['Seasonal Difference'] = df['Milk in Pounds per Cow'] - df['Milk in Pounds per Cow'].shift(12)


# In[40]:


adf_check(df['Seasonal Difference'].dropna())


# In[41]:


df['Seasonal First Difference'] = df['First Difference'] - df['First Difference'].shift(12)


# In[42]:


adf_check(df['Seasonal First Difference'].dropna())


# In[43]:


# Step 6. Auto correlation plot.


# In[44]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[45]:


fig_first = plot_acf(df['First Difference'].dropna())


# In[47]:


fig_seasonal_first = plot_acf(df['Seasonal First Difference'].dropna())


# In[48]:


from pandas.plotting import autocorrelation_plot


# In[49]:


autocorrelation_plot(df['Seasonal First Difference'].dropna())


# In[50]:


result = plot_pacf(df['Seasonal First Difference'].dropna())


# In[51]:


plot_acf(df['Seasonal First Difference'].dropna())
plot_pacf(df['Seasonal First Difference'].dropna())


# In[52]:


# Step 7. Fit ARIMA model


# In[53]:


from statsmodels.tsa.arima_model import ARIMA


# In[57]:


model = sm.tsa.statespace.SARIMAX(df['Milk in Pounds per Cow'],order=(0,1,0),seasonal_order=(1,1,1,12))


# In[62]:


results = model.fit()


# In[63]:


results.resid.plot()


# In[64]:


results.resid.plot(kind='kde')


# In[66]:


df['forecast'] = results.predict(start=150,end=168)
df[['Milk in Pounds per Cow','forecast']].plot(figsize=(12,8))


# In[68]:


# Step 8. Forecast real values.
from pandas.tseries.offsets import DateOffset


# In[69]:


future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1,24)]


# In[72]:


future_df = pd.DataFrame(index=future_dates,columns=df.columns)


# In[74]:


final_df = pd.concat([df,future_df])


# In[76]:


final_df['forecast'] = results.predict(start=168,end=192)


# In[83]:


final_df[['Milk in Pounds per Cow','forecast']].plot(figsize=(16,8))

