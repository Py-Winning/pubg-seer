
# coding: utf-8

# # PUBGGGGG
# #drop rankPoints, killStreaks, longestKill, matchId, roadKills, vehicleDestroys, weaponsAcquired, 
# #one-hot encoding matchType
# #One USer has 64 kills and 1.0 win percentage
# #Combine distances (walk, swim, and rideDistance) by adding them for totalDistance
# #Take into account afks

# In[57]:


get_ipython().magic(u'matplotlib inline')

print(__doc__)

import numpy as np
import scipy as sp
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split # Used to split the dataset effeciently
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO

#read the csv into a dataframe
df = pd.read_csv('Documents/pubg-seer/pubgdataset/train_V2.csv')


# In[58]:


df.head()


# In[59]:


df.count()


# In[60]:


#Drops all objects with NAs
df2 = df.dropna()
df2.count()


# In[61]:


#Combines walkDistance, swimDistance, and rideDistance into totalDistance
df2['totalDistance'] = df2['walkDistance'] + df2['swimDistance'] + df2['rideDistance']
df2.head()


# In[62]:


#Drops walkDistance, rideDistance, and swimDistance attributes
df3 = df2.drop(columns=['walkDistance', 'rideDistance', 'swimDistance'])
df3.head()


# In[63]:


#One hot encodes matchType
one_hot = pd.get_dummies(df3['matchType'])


# In[64]:


df3 = df3.drop('matchType',axis = 1)


# In[65]:


df3 = df3.join(one_hot)
df3


# In[66]:


df3.head()


# In[73]:


df4 = df3.copy()


# In[74]:


y = df4.pop('winPlacePerc').values
X = df4.values


# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

