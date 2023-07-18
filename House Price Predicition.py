#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Task to be performed

Task 1: Data Understanding: Begin by exploring the dataset and understanding its structure, 
        including the meaning and type of each feature.

Task 2: Data Cleaning: Perform data cleaning tasks to handle missing values, outliers, 
        and inconsistencies in the dataset.

Task 3: Feature Engineering: Perform feature engineering to enhance the predictive power 
        of the dataset. This may include creating new features, transforming existing features,
        or selecting relevant features.

Task 4: Data Preprocessing: Prepare the cleaned dataset for model training.
        This involves scaling numerical features, encoding categorical variables,
        and splitting the data into training and testing sets.

Task 5: Model Training and Evaluation: Choose an appropriate regression model 
        (e.g., linear regression, random forest, or gradient boosting) and 
        train it on the preprocessed dataset. Evaluate the model's performance using suitable
        metrics like mean squared error (MSE) or root mean squared error (RMSE).
Task 6: Model Optimization: Fine-tune the hyperparameters of the chosen model to
        improve its performance. You can use techniques like cross-validation or 
        grid search to find the best parameter values.

Task 7: Model Deployment: Once you have a satisfactory model, deploy it to make predictions
        on new, unseen data. You can use the trained model to predict house prices for new 
        instances and assess its real-world applicability.

Task 8: Linkedin Post: Once you complete all the above tasks, make a linkedin post from your 
        account for the entire Final Assignment completion.


# In[3]:


#Task 1: Data Understanding

#Importing required libraries
import pandas as pd   # for Dataframe manipulation
import numpy as np    # for numerical calculation
import seaborn as sns #data visualization
import matplotlib.pyplot as plt
import math

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
 
from scipy import stats
from scipy.stats import norm, skew


# In[4]:


# Loading the Data set
data=pd.read_csv('data.csv')


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.describe().T


# In[9]:


#Task 2: Data Preprocessing & Cleaning
#Converting dtypes of columns
data['date'] = pd.to_datetime(data['date'])
data['price'] = data['price'].astype('int64')
data['bedrooms']= data['bedrooms'].astype('int64')
data['bathrooms'] = data['bathrooms'].astype('int64')
data['floors']    = data['floors'].astype('int64')
data['street']    = data['street'].astype('string')
data['city']      = data['city'].astype('string')
data['statezip']  = data['statezip'].astype('string')
data['country']   = data['country'].astype('string')


# In[10]:


data.head().T


# In[11]:


data.info()


# In[12]:


data.insert(1, "year", data.date.dt.year)
data.head(3)


# In[13]:


data1 =data.drop("date", axis='columns')


# In[14]:


data1.head()


# In[15]:


## Handling missing, duplicate and 0 values
## Check for missing or null values
data1.isnull().sum()
#there is no missing values


# In[16]:


## Removing duplicate rows from the dataframe if there are in the data
data1.drop_duplicates()
data1.shape


# In[17]:


## Removing rows having price values 0

# Checking price having 0 values
price_zero = (data1.price == 0).sum()
print(price_zero)

# drop the column having price value 0
data1['price'].replace(0, np.nan, inplace = True)
data1.dropna(inplace=True)

# Checking shape of the dataset
print(data1.shape)


# In[18]:


## Dropping unnecessary columns from the dataset
data1 = data1.drop(['street'], axis = 1)
data1.head()


# In[19]:


## Number of unique value  in the dataset
data1.nunique(axis = 0)


# In[20]:


## Treating 'statezip' column and extracting the numeric code only
data1['statezip'] = data1['statezip'].str.split().str[1]
data1.head()


# In[21]:


# Reshape the column to a 2D array with a single feature
data1['statezip'] = np.reshape(data1['statezip'].values, (-1, 1))
data1.head()


# In[22]:


data1.info()


# In[23]:


## Converting dtypes of columns
data1['price']     = data1['price'].astype('int64')
data1['statezip'] = data1['statezip'].astype('int64')
data1['floors']    = data1['floors'].astype('int64')
data1.head()


# In[24]:


data1.info()


# In[25]:


## Univariate & Bivariant analysis of categorical columns
plt.figure(figsize=(15, 30))

plt.subplot(6, 2, 1)
pd.value_counts(data1['bedrooms']).plot(kind='bar')
plt.title("Bedroom Counts")

plt.subplot(6, 2, 2)
sns.barplot(x = data1['bedrooms'], y = data1.price)
plt.title("Bedroom - Price")

plt.subplot(6, 2, 3)
pd.value_counts(data1['bathrooms']).plot(kind='bar')
plt.title("Bathroom Counts")

plt.subplot(6, 2, 4)
sns.barplot(x = data1['bathrooms'], y = data1.price)
plt.title("Bathroom - Price")

plt.subplot(6, 2, 5)
pd.value_counts(data1['floors']).plot(kind='bar')
plt.title("Floor Counts")

plt.subplot(6, 2, 6)
sns.barplot(x = data1['floors'], y = data1.price)
plt.title("Floor - Price")

plt.subplot(6, 2, 7)
pd.value_counts(data1['waterfront']).plot(kind='bar')
plt.title("Waterfront Counts")
plt.subplot(6, 2, 8)
sns.barplot(x = data1['waterfront'], y = data1.price)
plt.title("waterfront - Price")

plt.subplot(6, 2, 9)
pd.value_counts(data1['view']).plot(kind='bar')
plt.title("View Counts")

plt.subplot(6, 2, 10)
sns.barplot(x = data1['view'], y = data1.price)
plt.title("View - Price")

plt.subplot(6, 2, 11)
pd.value_counts(data1['condition']).plot(kind='bar')
plt.title("Condition Counts")

plt.subplot(6, 2, 12)
sns.barplot(x = data1['condition'], y = data1.price)
plt.title("Condition - Price")

plt.show()


# In[26]:


## Lets visualize the outliers using box-plot
sns.boxplot(data1['price'])
plt.title('Outliers present in the Data')
plt.show()


# In[27]:


## Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = data1['price'].quantile(0.25)
Q3 = data1['price'].quantile(0.75)

## Calculate the interquartile range (IQR)
IQR = Q3 - Q1

## Define the lower and upper bounds for outlier detection
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

## Find the outliers in the column
outliers = data1[(data1['price'] < lower_bound) | (data1['price'] > upper_bound)]

## Count the number of outliers
outlier_count = len(outliers)

## Print the number of outliers
print("Number of outliers in 'price' column :",outlier_count)
outliers.head()


# In[28]:


# Convert the outliers to NaN
data1['price'][outliers.index] = np.nan
data1.head(2)


# In[29]:


## Lets check again the total number of missing values in the 'price' column
data1['price'].isnull().sum()


# In[30]:


## Fill the NaN values with the mean

# Calculate the mean value (rounded to 0 decimal places)
mean_value = round(data1['price'].mean())

# Fill null values with the rounded mean value
data1['price'].fillna(mean_value, inplace=True)
#df['price'] = df['price'].fillna(df['price'].mean())
data1.head(3)


# In[31]:


## Again checking distribution of price
sns.distplot(data1['price'],color="#2E4F4F",kde=True)
plt.show()


# In[ ]:


Task 3: Feature Engineering

A. Log Transformation on Target Variable
With the help of Q-Q plot we see whether the target variable is Normally Distributed or not, 
as Linear mostly like Normally Distributed Data.


# In[32]:


## Plotting QQ-plot
fig = plt.figure()
res = stats.probplot(data1['price'], plot=plt)
plt.show()


# In[35]:


##As the target variable (price) is very skewed, so we apply log-transformation on target varibale to make it Normally Distributed.
## Applying log-transformation
data1['price'] = np.log(data1['price'])

## Again plotting QQ-plot
fig = plt.figure()
res = stats.probplot(data1['price'], plot=plt)
plt.show()


# In[36]:


## Checking distribution of price again
sns.distplot(data1['price'],color="#FF8400",kde=True,fit=norm)


# In[37]:


## Creating heatmap to check the correlation in the dataset
plt.rcParams['figure.figsize'] = (12,12)

sns.heatmap(data1.corr(), annot=True)
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()


# In[39]:


###Encoding Independent Variables

## Applying encoding on columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'city' column
data1['city'] = label_encoder.fit_transform(data1['city'])
data1['country'] = label_encoder.fit_transform(data1['country'])
data1['bedrooms'] = label_encoder.fit_transform(data1['bedrooms'])
data1['bathrooms'] = label_encoder.fit_transform(data1['bathrooms'])
data1['price'] = label_encoder.fit_transform(data1['price'])
data1['sqft_living'] = label_encoder.fit_transform(data1['sqft_living'])
data1['sqft_lot'] = label_encoder.fit_transform(data1['sqft_lot'])
data1['sqft_above'] = label_encoder.fit_transform(data1['sqft_above'])
data1['sqft_basement'] = label_encoder.fit_transform(data1['sqft_basement'])
data1['yr_built'] = label_encoder.fit_transform(data1['yr_built'])
data1['yr_renovated'] = label_encoder.fit_transform(data1['yr_renovated'])


# In[41]:


data1.columns


# In[42]:


## Creating heatmap to check the correlation in the dataset
plt.rcParams['figure.figsize'] = (12,12)
sns.heatmap(data1.corr(), annot=True, cmap="Blues")
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()


# In[43]:


data1.columns


# In[45]:


data1.dtypes


# In[46]:


###Task 5: Model Training and Evaluation

##Splitting the Data and Target
## Creating dependent and independent sets
X = data1.drop(['price',], axis = 1)
Y = data1['price']

print(X.head())
print(Y.head())
   


# In[47]:


## Perform train test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[49]:


# Feature Scaling
## Perform standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler
StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Model Training
### Creating Models using diffenet algorithms

## 1. Creating a  Linear Regression Model
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train, Y_train)
Y_pred1 = LR.predict(X_test)


# In[50]:


## 2. Creating a Random Forest Model
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
Y_pred2 = RF.predict(X_test)


# In[51]:


## 3. Creating a Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor

GB = GradientBoostingRegressor()
GB.fit(X_train, Y_train)
Y_pred3 = GB.predict(X_test)


# In[52]:


## 4. Creating a SVR Model
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, Y_train)
Y_pred4 = svr.predict(X_test)


# In[53]:


## 5. Creating a Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor()
DT.fit(X_train, Y_train)
Y_pred5 = DT.predict(X_test)


# In[56]:


## 6. Creating Ridge Model
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, Y_train)
Y_pred6 = ridge.predict(X_test)


# In[ ]:


## Evaluation Metrics
#R2 Score
#Root Mean Squared Error (RMSE)
#Mean Squared Error (MSE)


# In[58]:


## Checking Model accuracy
from sklearn.metrics import r2_score, mean_squared_error

r2_score1 = r2_score(Y_test, Y_pred1)
r2_score2 = r2_score(Y_test, Y_pred2)
r2_score3 = r2_score(Y_test, Y_pred3)
r2_score4 = r2_score(Y_test, Y_pred4)
r2_score5 = r2_score(Y_test, Y_pred5)
r2_score6 = r2_score(Y_test, Y_pred6)

mse1 = mean_squared_error(Y_test, Y_pred1)
mse2 = mean_squared_error(Y_test, Y_pred2)
mse3 = mean_squared_error(Y_test, Y_pred3)
mse4 = mean_squared_error(Y_test, Y_pred4)
mse5 = mean_squared_error(Y_test, Y_pred5)
mse6 = mean_squared_error(Y_test, Y_pred6)

rmse1 = np.sqrt(mse1)
rmse2 = np.sqrt(mse2)
rmse3 = np.sqrt(mse3)
rmse4 = np.sqrt(mse4)
rmse5 = np.sqrt(mse5)
rmse6 = np.sqrt(mse6)


print("Linear Regression R2 Score :", r2_score1)
print("Linear Regression MSE :", mse1)
print("Linear Regression RMSE :", rmse1)
print("Random Forest R2 Score :", r2_score2)
print("Random Forest MSE :", mse2)
print("Random Forest RMSE :", rmse2)
print("Gradient Boosting R2 Score :", r2_score3)
print("Gradient Boosting MSE :", mse3)
print("Gradient Boosting RMSE :", rmse3)
print("SVR R2 Score :", r2_score4)
print("SVR MSE :", mse4)
print("SVR RMSE :", rmse4)
print("Decision Tree R2 Score :", r2_score5)
print("Decision Tree MSE :", mse5)
print("Decision Tree RMSE :", rmse5)
print("Ridge R2 Score :", r2_score6)
print("Ridge MSE :", mse6)
print("Ridge RMSE :", rmse6)


# In[59]:


## Checking scores of the models
print(LR.score(X_test,Y_test),": Linear Regression")
print(RF.score(X_test,Y_test),": Random Forest")
print(GB.score(X_test,Y_test),": Gradient Boosting")
print(svr.score(X_test,Y_test),": SVR")
print(DT.score(X_test,Y_test),": Decision Tree")
print(ridge.score(X_test,Y_test),": Ridge")


# In[62]:


#Task 6: Model Optimization

#A. Cross Validaion
# Calculate the train and test scores of XGBRegressor model
train_score =DT.score(X_train, Y_train)
test_score = DT.score(X_test, Y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)


# In[63]:


## Performing cross validaion
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DT, X, Y, cv=10)
print(scores)
print("Mean of all scores: ",scores.mean())


# In[65]:


#B. Evaluating the Algorithms
## Creating dataframe for Models with scores
final_data = pd.DataFrame({'Models':['Linear Regression', 'Random Forest Regressor', 
                'Gradient Boosting Regressor', 'SVR', 'DecisionTreeRegressor', 'ElasticNet'], 
                        'R2_Score': [r2_score1, r2_score2, r2_score3, r2_score4, r2_score5, r2_score6]})

models_df = pd.DataFrame(final_data)

# Sort the DataFrame based on R2_Score in descending order
models_df_sorted = models_df.sort_values(by='R2_Score', ascending=False)

# Apply background gradient to the R2_Score column
models_df_sorted_styled = models_df_sorted.style.background_gradient(subset=['R2_Score'], cmap='Blues')
models_df_sorted_styled


# In[66]:


## Visualize the scores on barplot
plt.style.use('seaborn')
plt.figure(figsize = (10, 5))
sns.barplot(final_data['Models'],final_data['R2_Score'])

# Set the axis labels and title
plt.xlabel('Models', fontsize= 14)
plt.ylabel('R2 Scores', fontsize= 14)
plt.title('Comparison of R2 Scores', fontsize = 18)
plt.xticks(fontsize= 12, rotation = 90)
plt.yticks(fontsize= 12, rotation = 45)
plt.show()


# In[68]:


#Task 7: Model Deployment Steps

#A. Creating Pickle File
import pickle 
pickle.dump(RF,open("model_rf.pkl", 'wb'))
#Save processed data as a CSV file
data.to_csv('processed_data.csv', index=False)


# In[ ]:




