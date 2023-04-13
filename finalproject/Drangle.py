#!/usr/bin/env python
# coding: utf-8

# ## Data Rangling 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


indicators= pd.read_csv ("CaribIndicators.csv")
policy= pd.read_csv("Mat_ChildPolicyCaribb.csv")


# In[9]:


countrytrends= indicators[["Year","Country","FemPop","GdpCap","Wrep_age15_49","IncomeGroup","Avg_MMR"]].dropna()


# In[4]:


indi_95 =indicators.query("Year==1995")
indi_05 =indicators.query("Year==2005").dropna()
indi_15 =indicators.query("Year==2015").dropna()


# In[5]:


indi_95= indi_95.drop(columns=["Prev_Anae","Avg_MMR"]).dropna()


# In[6]:


mch_95= indi_95.merge(policy,how = "left",
                          left_on="Country", right_on="Country")


# In[7]:


mch_05=indi_05.merge(policy,how = "left",
                          left_on="Country", right_on="Country")
mch_15=indi_15.merge(policy,how = "left",
                          left_on="Country", right_on="Country")


# In[ ]:




