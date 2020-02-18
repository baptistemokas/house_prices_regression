#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor
get_ipython().run_line_magic('matplotlib', 'inline')

orders = pd.read_csv ('./data/Orders.csv')


# In[66]:


returns = pd.read_csv('./data/Returns.csv')


# In[31]:


orders.Profit = pd.to_numeric(orders['Profit'].str.replace('$', '').str.replace(',', ''))
orders.Sales = pd.to_numeric(orders['Sales'].str.replace('$', '').str.replace(',', ''))


# In[32]:


orders['Ship.Date'] = pd.to_datetime(orders['Ship.Date'])


# In[33]:


orders['Order.Date'] = pd.to_datetime(orders['Order.Date'])


# In[65]:


x = orders.groupby('Ship.Date')['Ship.Date']
y = orders.groupby('Ship.Date')['Quantity'].sum()

plt.plot(x,y)


# In[78]:


new_returns = orders.merge(returns, how='left', on='Order.ID')


# In[89]:


new_returns[(new_returns.Returned == 'Yes') & (new_returns.Profit > 0)].Profit.sum() # lost $98203.35 in profits to returns


# In[24]:


orders.columns


# In[34]:


orders


# In[68]:


returns


# In[72]:


returns['Returned'].isna().sum()


# In[75]:


returns = returns.rename(columns={"Order ID": "Order.ID"})


# In[76]:


returns


# In[ ]:




