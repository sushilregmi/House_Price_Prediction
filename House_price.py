#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[62]:


train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")


# **Data Cleaning**

# In[63]:


#Calculating missing value count in each attribute.
missing = train.isnull().sum().sort_values(ascending = False)
missing.head(20)


# In[62]:


sns.heatmap(train.isnull(),yticklabels= False,cbar=False)


# **Calculating Correlation**

# In[63]:


#Calculating correlation of each attributes with target variable 'SalePrice'
correlation = train.corr()
print(correlation['SalePrice'].sort_values(ascending =  False),'\n')


# In[64]:


#Plotting correlation heatmap.
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,cmap= 'Blues',vmax=.8, square=True);


# In[84]:


#Plotting 10 highly correlated attributes with 'SalePrice'.
k = 10
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm,vmax=0.8,linewidth=0.01,square = True, annot = True,cmap='Blues',linecolor = 'white',xticklabels=cols.values,annot_kws={'size':12},yticklabels =cols.values)


# In[65]:


#Plotting scatter plot.
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], height = 4)
plt.show();


# In[67]:


#Box plot of 'SalePrice'
sns.boxplot(x=train['SalePrice'])


# **Merging train and test data for cleaning and transformations**

# In[69]:


#Concatenate train and test data to obtain data DataFrame.
target = train['SalePrice']
ids = test['Id']

train = train.drop(['Id','SalePrice'],axis = 1)
test =  test.drop('Id',axis =1 )

data = pd.concat([train, test], axis =0 ).reset_index(drop = True)
data.columns


# **Feature Selection**
# *  Attributes with low correlation with target variable 'SalePrice' are removed. *
# *  Removal of multicollinear attributes. *
# *  Attributes with more than 50 % of missing values are removed. *

# In[70]:


remove_columns =['GarageArea','TotalBsmtSF','TotRmsAbvGrd','GrLivArea','FireplaceQu','Fence','Alley','MiscFeature','PoolQC','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']


# In[71]:


data.drop(remove_columns,axis=1,inplace=True)


# In[72]:


data.columns


# In[75]:


#filling the values of missing categorical features
data.select_dtypes(object).columns


# In[76]:


#Filling missing values in which NA specifies to actual category value.
for column in ['BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish','GarageQual','GarageCond']:
    data[column] = data[column].fillna('None')
#filling missing values in which there is actual missing value with mode value.
for column in ['MSZoning','Utilities','Exterior1st', 'Exterior2nd', 'MasVnrType','Electrical','KitchenQual','Functional','SaleType']:
    data[column] = data[column].fillna(data[column].mode()[0])


# In[77]:


#checking whether the categorical attributes have missing value or not.
data.select_dtypes('object').isna().sum()


# In[87]:


#filling numeral missing values with mean of respective columns.
for column in ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
       'BsmtFullBath', 'GarageYrBlt',
       'GarageCars']:
    data[column] = data[column].fillna(data[column].mean())


# In[88]:


#finding attributes with high skewness
skew = pd.DataFrame(data.select_dtypes(np.number).columns, columns=['Feature'])
skew['Skew'] = skew['Feature'].apply(lambda feature: scipy.stats.skew(data[feature]))
skew['Absolute_Skew'] = skew['Skew'].apply(abs)
skew['is_Skew'] = skew['Absolute_Skew'].apply(lambda x: True if x >= 0.5 else False)
skew


# In[90]:


#log transformation of skewed data
for column in skew.query("is_Skew == True")['Feature'].values:
    data[column] = np.log1p(data[column])


# In[83]:


#encoding of categorical data
data = pd.get_dummies(data)
data


# In[84]:


#Scaling
scaler= StandardScaler()
scaler.fit(data)

data =pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
data


# In[82]:


target.hist()


# In[83]:


#visualization of log transformation of target variable
plt.figure(figsize=(20, 10))
sns.histplot(target, kde=True)
plt.title("Without Log Transform")
sns.displot(np.log1p(target), kde=True)
plt.xlabel("Log SalePrice")
plt.title("With Log Transform")

plt.show()


# In[84]:


#Splitting train and test
train = data.loc[:train.index.max(),:]
test = data.loc[train.index.max()+1:,:]


# In[85]:


#log transformation of target variable
log_target = np.log1p(target)


# **Model Selection**

# In[92]:


#define function for calculating metrics R squared score and RMSE score.
def rsqr_score(test, pred):
    r_sqr = r2_score(test, pred)
    return r_sqr

def rmse_score(test, pred):
    rmse = np.sqrt(mean_squared_error(test, pred))
    return rmse

def print_score(test, pred, model):
    print(f"- Regression: {model}")
    print(f"R²: {rsqr_score(test, pred)}")
    print(f"RMSE: {rmse_score(test, pred)}\n")
    


# In[87]:


#Train test split
X_train, X_test, y_train, y_test = train_test_split(train,log_target , test_size = 0.2, random_state = 42)


# In[88]:


print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
print(f"\nX_test:{X_test.shape}\ny_test:{y_test.shape}")


# **Ridge Regression**

# In[93]:


ridge_model = Ridge(alpha = 0.001)
scores_ridge = -1 * cross_val_score(ridge_model, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores:\n", scores_ridge)
print("Mean MSE scores:", scores_ridge.mean())


# In[94]:


ridge_model.fit(X_train, y_train)
y_pred_r = ridge_model.predict(X_test)
print_score(y_test, y_pred_r, "Ridge")


# In[95]:


plt.figure()
plt.title("Ridge", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_r),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# **Lassso Regression**

# In[47]:


lasso_model = Lasso(alpha = 0.001)


# In[96]:


scores_lasso = -1 * cross_val_score(lasso_model, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores:\n", scores_lasso)
print("Mean MSE scores:", scores_lasso.mean())


# In[97]:


lasso_model.fit(X_train, y_train)
y_pred_l = lasso_model.predict(X_test)
print_score(y_test, y_pred_l, "Lasso")


# In[98]:


plt.figure()
plt.title("Lasso", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_l),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# **Random Forest Regression**

# In[99]:


ran_model = RandomForestRegressor()


# In[100]:


scores_ran = -1 * cross_val_score(ran_model, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_squared_error')

print("MSE scores:\n", scores_ran)
print("Mean MSE scores:", scores_ran.mean())


# In[101]:


ran_model.fit(X_train, y_train)
y_pred_ran = ran_model.predict(X_test)
print_score(y_test, y_pred_ran, "Random Forest")


# In[103]:


plt.figure()
plt.title("Random Forest", fontsize=20)
plt.scatter(np.exp(y_test), np.exp(y_pred_ran),
            color="deepskyblue", marker="o", facecolors="none")
plt.plot([0, 800000], [0, 800000], "darkorange", lw=2)
plt.xlabel("\nActual Price", fontsize=16)
plt.ylabel("Predicted Price\n", fontsize=16)
plt.show()


# **Prediction on test data**

# In[104]:


#predicting on a test data
y_pred = np.exp(lasso_model.predict(test))

output = pd.DataFrame({"Id": ids,
                       "SalePrice": y_pred})

output.to_csv("Prediction.csv")
output


# In[ ]:




