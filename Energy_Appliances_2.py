#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import zipfile
import time
import dateutil
import sklearn.metrics as sm
from sklearn import preprocessing, model_selection, metrics
from sklearn.metrics import accuracy_score


# In[4]:


data = pd.read_csv("KAG_energydata_complete.csv")


# In[5]:


data.head(5)


# In[6]:


#To check null values
data.isnull().sum().sort_values(ascending=False)


# In[7]:


#Check unique values in each features
data.apply(lambda x: len(x.unique()))


# In[8]:


#Check the statistical analysis of the dataset
data.describe()


# In[9]:


#To check the datatype of all the features
data.info()


# In[10]:


# starting of EDA
# Data Visualization part


# In[11]:


sns.displot(data["Appliances"])


# In[12]:


#2-Clearly there will be outliers in appliances part as it has a skew graph so we will use box plot to visualize that
sns.boxplot(x= data["Appliances"], color="green")


# In[13]:


# #so in above box plot we can see the outliers
# filter = data["Appliances"].values < 125
# data_outlier_rem = data[filter]
# data_outlier_rem


# In[14]:


#sns.boxplot(x= data_outlier_rem["Appliances"], color="green")


# In[15]:


#data = data_outlier_rem


# In[16]:


# Histogram of all the features to understand the distribution
data.hist(bins = 20 , figsize= (12,16)) ;


# In[17]:


# Observations based on distribution plot
# 1. All humidity values except RH_6 and RH_out follow a Normal distribution.
# 2. Similarly, all temperature readings follow a Normal distribution except for T9.
# 3. Out of the remaining columns, we can see that Visibility, Windspeed and Appliances are skewed.
# 4. The random variables rv1 and rv2 have more or less the same values for all the recordings.
# 5. The output variable Appliances has most values less than 200Wh, showing that high energy consumption cases are very low.
# 6. No column has a distribution like the target variable Appliances.
# Hence, there are no feature independent feature with a linear relationship with the target.


# In[18]:


#To include time from date column for analysis of appliances energy prediction from time and day point of view
data["exact_date"]=data['date'].str.split(' ').str[0]

data["hours"]=(data['date'].str.split(':').str[0].str.split(" ").str[1]).astype(str).astype(int)
data["seconds"]=((data['date'].str.split(':').str[1])).astype(str).astype(int).mul(60)

data["week"]=(data['date'].str.split(' ').str[0])
data["week"]=(data['week'].apply(dateutil.parser.parse, dayfirst=True))
data["weekday"]=(data['week'].dt.dayofweek).astype(str).astype(int)
data["week"]=(data['week'].dt.day_name())

data['log_appliances'] = np.log(data.Appliances)
data['hour*lights'] = data.hours * data.lights
data['hour_avg'] = list(map(dict(data.groupby('hours')["Appliances"].mean()).get, data.hours))

data.head(5)


# In[19]:


# Perform analysis & model development
# Day wise Electricity consumption
dates=data["exact_date"].unique()
arranged_day = pd.Categorical(data["exact_date"], categories=dates,ordered=True)
date_series = pd.Series(arranged_day)
table = pd.pivot_table(data,values="Appliances",index=date_series, aggfunc=[np.sum],fill_value=0)
table.plot(kind="bar",figsize=(20, 7))
plt.show()


# In[20]:


days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
arranged_day = pd.Categorical(data["week"], categories=days,ordered=True)
day_series = pd.Series(arranged_day)
table = pd.pivot_table(data,index=["hours"],
               values="Appliances",columns=day_series,
               aggfunc=[np.sum],fill_value=0)

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_title('Heatmap : Appliances(wh)')

heatmap = ax.pcolor(table)

ax.set_xlabel("Week Days")
ax.set_ylabel("Hours")

plt.colorbar(heatmap)
ax.set_yticks(range(len(table.index)+1))
ax.set_xticks(range(len(table.columns)+1))

plt.xlabel("Week")
plt.ylabel("Hours of Day")
plt.show()


# In[21]:


table.plot.box(figsize=(20, 7))


# In[22]:


#Weekends are observed to have high values of energy.This explains the outliers in appliances. 
# hence removing these outliers will not give any benifit to us as these are weekend effects


# In[23]:


# Plot of Mean Energy Consumption per Hour of a Day

data.groupby('hours')['Appliances'].mean().plot(figsize=(10,8))
plt.xlabel('Hour')
plt.ylabel('Appliances consumption in Wh')
ticks = list(range(0, 24, 1))
plt.title('Mean Energy Consumption per Hour of a Day')

plt.xticks(ticks);


# In[24]:


#High electricity consumption is observed during evening hours between 16:00 to 20:00
#at night hours from 23:00 to  6:00, the power load is below 50Wh which is quite obvious as most appliances at this time will be off or on standby
# between  9 to 13, consumption is > 100wh as it is breakfast and after that the consumpition is less than 100Wh.                                                                                           


# In[26]:


# now as the appliances energy consumption graph was skewed therefore i thought of logging that function so that 
# we regression models give a better fit to that. As when we take the log of the data and it becomes normalish, 
# then you can take advantage of many features of a normal distribution, like well-defined mean, standard deviation 
# (and hence z-scores), symmetry, etc. 

f, axes = plt.subplots(1, 2,figsize=(10,4))

sns.distplot(data["Appliances"],hist = False, color = 'blue',ax=axes[0])
axes[0].set_title("Appliance's consumption")
axes[0].set_xlabel('Appliances wH')

sns.distplot(data["log_appliances"],hist = False, color = 'blue',ax=axes[1])
axes[1].set_title("Log Appliance's consumption")
axes[1].set_xlabel('Appliances log(wH)')


# In[27]:


# Correlation among the variables
col = ['Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4',
       'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9',
       'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility',
       'Tdewpoint','hours']
corr = data[col].corr()
plt.figure(figsize = (18,18))
sns.set(font_scale=1)
sns.heatmap(corr, cbar = True, annot=True, square = True,cmap="RdYlGn", fmt = '.2f', xticklabels=col, yticklabels=col)
plt.show();


# In[25]:


#The energy consumpyion is highly correlated with
# Hours : 0.22
# Lights : 0.20
# T2 : 0.12
# T6 : 0.12


# In[29]:


# Correlation among the variables while considering log Appliances
col = ['log_appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4',
       'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9',
       'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility',
       'Tdewpoint','hours']
corr = data[col].corr()
plt.figure(figsize = (18,18))
sns.set(font_scale=1)
sns.heatmap(corr, cbar = True, annot=True, square = True,cmap="RdYlGn", fmt = '.2f', xticklabels=col, yticklabels=col)
plt.show();


# In[30]:


#The energy consumpyion is highly correlated with
# Hours : 0.33
# Lights : 0.26
# T2 : 0.21
# T6 : 0.10

# this graph shows that independent variables are more closely related to log appliances 


# In[ ]:


# Now we will try to do some feature selectrion process for our dataset if we could find any
# before that we will do the splitting of the data into test and training set


# In[38]:


# Creation of train/test sets
from sklearn.model_selection import train_test_split

# 75% of the data is usedfor the training of the models and the rest is used for testing
train, test = train_test_split(data,test_size=0.25,random_state=40)


# In[48]:


# Combining all the data and seegregating like columns together
col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]

col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]

col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",
                "Windspeed","Visibility"] \
col_light = ["lights"]

col_randoms = ["rv1", "rv2"]

col_target = ["Appliances"]

feature_vars = train[col_temp + col_hum + col_weather + col_light + col_randoms ]
target_vars = train[col_target]

# col_target1 = ["log_appliances"]
# Seperate dependent and independent variables 

# feature_vars1 = train[col_temp + col_hum + col_weather + col_light + col_randoms ]
# target_vars1 = train[col_target]


# In[49]:


feature_vars.describe()


# In[50]:


#As we can see that 75% of lights counts to 0. Therefore taking its proper information is necessary
feature_vars.lights.value_counts()


# In[51]:


# Due to lot of zero enteries this column is of not much use 
feature_vars.drop("lights", axis=1, inplace=True);


# In[52]:


feature_vars.head(2)


# In[53]:


# Use the weather , temperature , applainces and random column to see the correlation so that we can see if 
# we can remove some other variable also

train_corr = train[col_temp + col_hum + col_weather +col_target+col_randoms]
corr = train_corr.corr()

# Mask the repeated values
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
  
f, ax = plt.subplots(figsize=(16, 14))

#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, annot=True, fmt=".2f" , mask=mask,)
    #Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
plt.show()


# In[36]:


#the conclusions from above are:-
# All the temperature variables from T1-T9 and T_out have positive correlation with the target Appliances .
# There is a high degree of correlation between T9 and T3,T5,T7,T8 
# also T6 & T_Out has high correlation(0.97)(both temperatures from outside) . 
# Hence T6 & T9 can be removed from training set as information provided by them can be provided by other fields.
# Weather attributes - Visibility, Tdewpoint, Press_mm_hg have low correlation values
# Random variables have no role to play


# In[54]:


#Split training dataset into independent and dependent varibales
train_X = train[feature_vars.columns]
train_y = train[target_vars.columns]


# In[55]:


#Split testing dataset into independent and dependent varibales
test_X = test[feature_vars.columns]
test_y = test[target_vars.columns]


# In[56]:


#As a result of feature selection and 
# Due to conlusion made above below columns are removed
train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)
# Due to conlusion made above below columns are removed
test_X.drop(["rv1","rv2","Visibility","T6","T9"], axis=1, inplace=True)


# In[57]:


train_X.columns


# In[58]:


test_X.columns


# In[59]:


#Scaling the concerned data
# Variables that are measured at different scales do not contribute equally to the model fitting & model learned function
# and might end up creating a bias. Thus, to deal with this potential problem feature-wise standardized (μ=0, σ=1) 
# is usually used prior to model fitting.

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

# Create test and training set by including Appliances column

train = train[list(train_X.columns.values) + col_target ]


test = test[list(test_X.columns.values) + col_target ]

# Create dummy test and training set to hold scaled values

sc_train = pd.DataFrame(columns=train.columns , index=train.index)

sc_train[sc_train.columns] = sc.fit_transform(train)

sc_test= pd.DataFrame(columns=test.columns , index=test.index)

sc_test[sc_test.columns] = sc.fit_transform(test)


# In[60]:


sc_train.head()


# In[61]:


sc_test.head()


# In[68]:


# Remove Appliances column from traininig set

train_X =  sc_train.drop(['Appliances'] , axis=1)
train_y = sc_train['Appliances']

test_X =  sc_test.drop(['Appliances'] , axis=1)
test_y = sc_test['Appliances']


# In[ ]:


# Trying out 4 Regression models:
# 1. LinearRegression
# 2. SVR
# 3. RandomForestRegressor
# 4. XG Boost 


# In[69]:


# Starting of modelling for the data
from sklearn import linear_model

lin_model = linear_model.LinearRegression()
lin_model.fit(train_X,train_y)


# In[70]:


from sklearn import svm

svr_model = svm.SVR(gamma='scale')
svr_model.fit(train_X,train_y)


# In[71]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100,random_state=1)            
rf_model.fit(train_X, train_y)


# In[72]:


import xgboost as xgb
from xgboost import plot_importance
model_xgb = xgb.XGBRegressor(objective='reg:squarederror')
model_xgb.fit(train_X, train_y)


# In[73]:


# metrics we are using
# 1- mape = mean absolute percentage error
# Mean absolute percentage error is commonly used as a loss function for regression problems and in model evaluation,
# because of its very intuitive interpretation in terms of relative error.
# MAPE = (1 / sample size) x ∑[( |actual - forecast| ) / |actual| ] x 100
# Accuracy will hence be  = 100 - MAPE


# 2- r2 Score = R2 score is used to evaluate the performance of a linear regression model. 
# It is the amount of the variation in the output dependent attribute which is predictable from the input 
# independent variable(s). It is used to check how well-observed results are reproduced by the model, 
# depending on the ratio of total deviation of results described by the model.
# Mathematical formula - R2= 1- SSres / SStot
# Where,
# SSres is the sum of squares of the residual errors.
# SStot is the total sum of the errors.


# In[75]:



from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from math import sqrt

# Function to evaluate the models
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    r_score = 100*r2_score(test_labels,predictions)
    accuracy = 100 - mape
    print(model,'\n')
    print('Average Error: {:0.4f} degrees'.format(np.mean(errors)))
    print('Variance score R^2  : {:0.2f}%' .format(r_score))
    print("Accuracy :{}%\n".format(accuracy))


# In[76]:


evaluate(lin_model, test_X, test_y)
evaluate(svr_model, test_X, test_y)
evaluate(rf_model, test_X, test_y)
evaluate(model_xgb, test_X, test_y)


# In[ ]:


#Now for fine tuning we will use cross Validation Technique


# In[77]:


#instead of KFold I use TimeSeriesSplit (10 splits) due to time series data
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits = 10)

print('Linear Model:')
scores = cross_val_score(lin_model, train_X, train_y, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(lin_model, train_X, train_y, cv=cv,scoring='r2')
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))

print('SVR Model:')
scores = cross_val_score(svr_model, train_X, train_y, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(svr_model, train_X, train_y, cv=cv)
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))

print('Random Forest Model:')
scores = cross_val_score(rf_model, train_X, train_y, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(rf_model, train_X, train_y, cv=cv)
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))

print('XGBRegressor Model:')
scores = cross_val_score(model_xgb, train_X, train_y, cv=cv,scoring='neg_mean_absolute_error')
print("Accuracy: %0.2f (+/- %0.2f) degrees" % (100+scores.mean(), scores.std() * 2))
scores = cross_val_score(model_xgb, train_X, train_y, cv=cv)
print("R^2: %0.2f (+/- %0.2f) degrees" % (scores.mean(), scores.std() * 2))


# In[78]:


# Model performance on test data
y1_pred = lin_model.predict(test_X)
y2_pred = svr_model.predict(test_X)
y3_pred = rf_model.predict(test_X)
y5_pred = model_xgb.predict(test_X)


# In[79]:


fig = plt.figure(figsize=(20,8))
plt.plot(test_y[:100].values,label='Target value',color='b')
plt.plot(y1_pred[:100],label='Linear Prediction ', linestyle='--', color='y')
plt.plot(y2_pred[:100],label='SVR Prediction ', linestyle='--', color='g')
plt.plot(y3_pred[:100],label='Random Forest', linestyle='--', color='r')
plt.plot(y5_pred[:100],label='XGB', linestyle='--', color='orange')

plt.legend(loc=1)


# In[ ]:


Observations:-
#random Forest is best algorithm for this data set
#more improvement which can be done on it
#log of appliances can be done bcoz logging changes the skew distribution into normal distribution.So linear regression model will give a much better accuracy with logging 


# In[ ]:




