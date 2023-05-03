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


# In[3]:


countrytrends= indicators[["Year","Country","FemPop","GdpCap","Wrep_age15_49","IncomeGroup","Avg_MMR"]].dropna()


# In[4]:


indi_95 =indicators.query("Year==1995")
indi_05 =indicators.query("Year==2005").dropna()
indi_15 =indicators.query("Year==2015").dropna()


# In[5]:


indi_95= indi_95.drop(columns=["Prev_Anae","Avg_MMR"]).dropna()


# In[6]:


summary95=indi_95.describe()
summary95


# In[7]:


summary05=indi_05.describe()
summary05


# In[8]:


summary15=indi_15.describe()
summary15


# In[9]:


mch_95= indi_95.merge(policy,how = "left",
                          left_on="Country", right_on="Country")


# In[10]:


mch_05=indi_05.merge(policy,how = "left",
                          left_on="Country", right_on="Country")
mch_15=indi_15.merge(policy,how = "left",
                          left_on="Country", right_on="Country")


# ### Pandas Analysis

# In[11]:


NrMR= pd.merge(indi_95, indi_05, on='Country').merge(indi_15, on=['Country'])
NrMR


# In[12]:


Neo_MR= NrMR[["Country","Neonat_MR_x","Neonat_MR_y","Neonat_MR"]].rename(columns={"Neonat_MR_x": "1995NrMR", "Neonat_MR_y": "2005NrMR","Neonat_MR": "2015NrMR"})


# In[13]:


Prev_Fe = NrMR[["Country","Prev_Anae_x","Prev_Anae_y"]].rename(columns={"Prev_Anae_x": "2005Fe","Prev_Anae_y": "2015Fe"})


# In[14]:


MMR_dec = NrMR[["Country","Avg_MMR_x","Avg_MMR_y"]].rename(columns={"Avg_MMR_x": "2005mmR","Avg_MMR_y": "2015mmR"})


# In[15]:


def abs_diff(df, col1, col2, col3=None):
    if col3 is None:
        df['abs_diff'] = df[col1].sub(df[col2]).abs()
    else:
        df['abs_diff'] = (df[col1].sub(df[col2]).abs() + df[col1].sub(df[col3]).abs() + df[col2].sub(df[col3]).abs()) / 3
    return df


# In[16]:


# calculate difference between Neonatal MR for 
MR_neo = abs_diff(Neo_MR, '1995NrMR', '2005NrMR','2015NrMR')
MR_neo


# In[52]:


def avg_GdpCap(df,df2, col1, col2, col3):
    df2['avg_GdpCap'] = (df[col1] + df[col2] + df[col3]) / 3
    return df2

Gdp_neo = avg_GdpCap(NrMR,MR_neo, 'GdpCap_x', 'GdpCap_y', 'GdpCap')
Gdp_neo["IncomeGroup"] = mch_15["IncomeGroup"]


# In[42]:


Gdp_neo['avg_NrMR'] = Gdp_neo[['1995NrMR','2005NrMR', '2015NrMR']].mean(axis=1)


# In[53]:


Gdp_neo


# In[17]:


Anaem_ = abs_diff(Prev_Fe,'2005Fe','2015Fe')
Anaem_.describe()


# In[39]:


Prev_Fe['avg_anemia'] = Prev_Fe[['2005Fe', '2015Fe']].mean(axis=1)


# ### Testing Impact of GDP on maternal and neonatal mortality

# In[31]:


import scipy.stats as stats

# Perform correlation test
corr, pval = stats.pearsonr(Gdp_neo ['abs_diff'], Gdp_neo ['avg_GdpCap'])

# Print results
print('Pearson correlation coefficient:', corr)
print('p-value:', pval)


# In[45]:


# Perform correlation test
corr, pval = stats.pearsonr(Gdp_neo ['avg_NrMR'], Gdp_neo ['avg_GdpCap'])

# Print results
print('Pearson correlation coefficient:', corr)
print('p-value:', pval)


# In[38]:


# Perform correlation test
corr, pval = stats.pearsonr(Prev_Fe['avg_anemia'], Gdp_neo ['avg_GdpCap'])

# Print results
print('Pearson correlation coefficient:', corr)
print('p-value:', pval)


# In[44]:


# Perform correlation test
corr, pval = stats.pearsonr(Gdp_neo ['avg_NrMR'], Prev_Fe['avg_anemia'])

# Print results
print('Pearson correlation coefficient:', corr)
print('p-value:', pval)


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Split data into features and labels
X = Gdp_neo.drop(['Country', 'IncomeGroup'], axis=1)
y = Gdp_neo['IncomeGroup']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and KNeighborsClassifier
pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])

# Define a range of values for n_neighbors to test
k_values = range(1, 10)

# Fit the pipeline and test accuracy for each value of n_neighbors
accuracy_scores = []
for k in k_values:
    pipeline.set_params(knn__n_neighbors=k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot the accuracy scores for each value of n_neighbors
plt.plot(k_values, accuracy_scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Performance Comparison')
plt.show()

# Find the value of n_neighbors that gives the highest accuracy
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print('Best k: {}'.format(best_k))


# In[55]:


accuracy_scores


# In[ ]:




