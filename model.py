#!/usr/bin/env python
# coding: utf-8

# Cloud = Server + Storage + Software Platform 
# 
# Cloud is very reliable and Secure
# 
# Servers and storage will store the data
# 
# Acccesing of the cloud will be done by Internet and Dedicated fibre networks 
# 
# Types of Clouds:
# 
#     1)Public cloud   ex: Google, Gmail
#     2)Private Cloud  ex: Google, Microsoft  ----- Company owned No public access
#     3) Hybrid Coud   ex : Adhar  ---- Used by Government employees
#     
# Application built on cloud:
# 
# Different service mopdels will be there
# 
# 1) IAAS : Infrastructure as a Servies
# 
#         rent(Server, Storage, Operating System,Virtual m/c's)
#         
# 2) PAAS : Platform as a Service
# 
#         Building Applications and software products and services
#         
#         Cloud provider will provide pre built services like Databases(SQL Based), WebApps (Microsoft Office suite, Google office Suite) and Email
# 
# 3) SAAS : Software as a service
# 
#         Rent Software like ERP-Oracle, SAP, Salesforce, Zoho, Dropbox or Google Drive, Payroll, Accounting Tally
#         
#         
# Clouds will be deployed in multiple geographical location
# 
#         A/c Cost is very high(checks for the location where heat generates less)
#         
#         Natural Desaters

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:





# In[4]:


df = pd.read_csv("Auto MPG Reg.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


# Convert horsepower into numeric

df.horsepower = pd.to_numeric(df.horsepower, errors="coerce")


# In[8]:


df.horsepower.describe()


# In[9]:


df.horsepower = df.horsepower.fillna(df.horsepower.median())


# In[10]:


y = df.mpg
X = df.drop(['carname','mpg'],axis=1)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


regmodel = LinearRegression().fit(X,y)


# In[13]:


regmodel.score(X,y)


# In[14]:


regpredict = regmodel.predict(X)


# In[15]:


from sklearn.metrics import mean_squared_error


# In[16]:


np.sqrt(mean_squared_error(y,regpredict))


# In[17]:


# For Deployment model needs to be saved as .pkl(pickle) file or .sav(joblib) library


# In[18]:


import joblib


# In[19]:


joblib.dump(regmodel,"reg.sav")


# In[ ]:




