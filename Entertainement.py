# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:06:46 2023

@author: hp

#Business objective-
user satisfaction in the entertainment domain by leveraging data analytics 
to optimize content curation.

#Minimize- Minimize user churn and enhance content discoverability to ensure a 
sustainable user base and consistent revenue in the entertainment sector.

#Maximize- Maximize user engagement and subscription retention through 
personalized content recommendations and interactive features.

#Constraint-1) Balance the need for personalized content with the requirement for 
a diverse and inclusive entertainment library.
2)Foster innovation in content creation and technology while managing costs 
effectively to maintain profitability.


#Data Dictionary

Feature Name	  Description	                                Type	        Relevance on Entertainment	                                                       Dataset columns

id	              Content ID assigned to each 
                  piece of entertainment content	            Identifier	    Enables efficient tracking and management of individual content items.	            id

title	          The title of the entertainment content	    Text	        Core identifier for each content item; used in user interfaces and recommendations.	title

category	      The genre or category classification 
                  of the entertainment content	                Categorical	    Facilitates content organization and user preferences based on genres.	            category

review	          Numeric or textual reviews/ratings given 
                  to the entertainment content              	Numeric/Text	Influences content recommendations and provides insights into user opinions.	     review


"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Entertainment dataset
df= pd.read_csv("C:/10-Recommandation engine/Entertainment.csv")
df

#in that dataset 51 rows and 4 columns

#show all columns in dataset
df.columns

'''It showing Id,Titles,category,reviews column,and
datatype is object.
'''

#in this dataset 51 are rows and 4 columns
df.shape


#it show 1st five rows
df.head()

#it show last 5 rows of dataset
df.tail()

#providing a statistical summary of the numerical columns in the DataFrame.
df.describe()

#check there is null value
df.isnull().sum()
#in that dataset no null values
'''Id          0
Titles      0
Category    0
Reviews     0
dtype: int64
'''

################################################

# Data Cleaning: Handling Missing Values
df=df.dropna(inplace=True)
#o/p - Remove rows with missing values..

###############################################

#summary
print(df.info())
'''class 'pandas.core.frame.DataFrame'>
RangeIndex: 51 entries, 0 to 50
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Id        51 non-null     int64  
 1   Titles    51 non-null     object 
 2   Category  51 non-null     object 
 3   Reviews   51 non-null     float64
dtypes: float64(1), int64(1), object(2)
'''
#show non null count and datatypes

# 4.2 Univariate Analysis
print("Univariate Analysis:")
# Histogram of Reviews
plt.figure(figsize=(8, 5))
sns.histplot(df['review'], bins=10, kde=True)
plt.title('Distribution of Reviews')
plt.xlabel('Review Scores')
plt.show()

#o/p - Displays histograms for numerical columns (e.g., 'review')

######################

# Countplot of Categories
plt.figure(figsize=(8, 5))
sns.countplot(x='category', data=df)
plt.title('Distribution of Categories')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.show()

#o/p - countplots for categorical columns (e.g., 'category')

#########################

# 4.3 Bivariate Analysis
print("Bivariate Analysis:")
# Scatter plot of Review Scores vs. Content ID
plt.figure(figsize=(8, 5))
sns.scatterplot(x='id', y='review', data=df)
plt.title('Scatter Plot of Review Scores vs. Content ID')
plt.xlabel('Content ID')
plt.ylabel('Review Scores')
plt.show()

#o/p - Shows scatter plots and boxplots to explore relationships between variables

############################

# Boxplot of Review Scores by Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='review', data=df)
plt.title('Boxplot of Review Scores by Category')
plt.xlabel('Category')
plt.ylabel('Review Scores')
plt.show()

#o/p - It provides a summary of the central tendency, spread, and potential outliers for each category.

##################################################

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Split the dataset into training and testing sets
trainset, testset = train_test_split(df, test_size=0.2, random_state=42)
'''
#o/p - in this we Split the dataset into training and testing sets.
'''

# Build the Recommender Engine model (User-based Collaborative Filtering)
sim_options = {'name': 'cosine', 'user_based': True}
model = KNNBasic(sim_options=sim_options)

'''
o/p - Use the KNNBasic collaborative filtering algorithm with cosine similarity.
'''

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)
'''
o/p - Train the model on the training set and evaluate its performance 
on the testing set using Root Mean Squared Error (RMSE).
'''

# Recommend items for a specific user (user with id 1)
user_id = 1
user_ratings = df.loc[df['id'] == user_id, ['id', 'user_rating', 'review']]
user_ratings = list(zip(user_ratings['id'], user_ratings['user_rating']))

# Get item recommendations for the user
item_recommendations = model.get_neighbors(user_id, k=2)

# Display the recommendations
recommended_items = df[df['id'].isin(item_recommendations)]
print(f"Recommended items for User {user_id}:")
print(recommended_items[['id', 'title', 'category', 'review']])

'''
o/p - in this we Demonstrate how to recommend items for a specific 
user (user with id 1) and display the recommended items.
'''
