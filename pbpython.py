
# coding: utf-8

# In[23]:


get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from fbprophet import Prophet

data_file = "googlemerchandisedata.xlsx"
df = pd.read_excel(data_file)
df.head()
print (df.head())


# In[24]:

print(df.dtypes)


# In[25]:

df.set_index('Date').plot();


# In[26]:

df.loc[(df['Sessions'] > 5000), 'Sessions'] = np.nan
df.set_index('Date').plot();


# In[27]:

df['Sessions'] = np.log(df['Sessions'])
df.set_index('Date').plot();


# In[28]:

df.columns = ["ds", "y"]
df.head()


# In[29]:

m1 = Prophet()
m1.fit(df)


# In[30]:

future1 = m1.make_future_dataframe(periods=365)


# In[31]:

forecast1 = m1.predict(future1)


# In[19]:

forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[32]:

np.exp(forecast1[['yhat', 'yhat_lower', 'yhat_upper']].tail())


# In[33]:

m1.plot(forecast1);


# In[34]:

m1.plot_components(forecast1);


# In[ ]:



