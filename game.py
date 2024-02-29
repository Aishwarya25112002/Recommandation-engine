# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:43:29 2023

@author: hp

       Recommandation Engine
       
#DataSet-Game

#Business Problem-
Business Objective
Maximize- Maximize player retention and in-game 
         spending by optimizing gameplay experiences, personalized content recommendations

Minimize-customer churn and negative feedback 
         through data-driven strategies that 
         enhance user satisfaction, leading to a 
         sustainable

Constaint-Adhere to data privacy regulations and 
          ensure the secure handling of player 
          information within the game dataset, 
          maintaining compliance with relevant legal frameworks.

Data Dictionary-

| Feature Name  | Description                                       | Type       | Relevance on Game Dataset                  |
|------------ --|---------------------------------------------------|------------|--------------------------------------------|
| userID        | Unique identifier for each player                | Numeric    | Key identifier for player data             |
| Game          | Unique identifier for each game                  | Numeric    | Key identifier for game data               |
| Rating        | Player's rating for the game (1 to 5 scale)      | Numeric    | Reflects player satisfaction with the game |

"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the game dataset
df= pd.read_csv("C:/Recommandation engine/game.csv")
df
#In that dataset 5000rows and 3 columns

df.columns #show the number of column in game dataset
#O/P-
'''Index(['userId', 'game', 'rating'], 
        dtype='object')'''

# 2.Data Cleaning and Data Mining
#Remove duplicates, handle missing values

df = df.drop_duplicates()
#In that code remove the Duplicate rows 

df = df.dropna() #that use to remove the rows for missing value 

# 4. Exploratory Data Analysis (EDA)

#Summary
print(df.info())
#it shows number of column 
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 5000 entries, 0 to 4999
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   userId  5000 non-null   int64  
 1   game    5000 non-null   object 
 2   rating  5000 non-null   float64
dtypes: float64(1), int64(1), object(1)
memory usage: 156.2+ KB
None
print(df.describe())
'''

#4.2 Univariate Analysis
#Plot distribution of playtime
plt.figure(figsize=(10,6))
sns.histplot(df['rating'], bins=30, kde=True)
plt.title('Distribution of Playtime')
plt.show()

#4.3 Bivariate Analysis
#Plot correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 5. Model Building

#Build Recommender Engine Model (Collaborative Filtering)
# Assume you have 'userID', 'GameID', and 'Rating' columns
ratings = df[['userId', 'game', 'rating']]

# Split data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_data, test_data

# Create a user-item matrix
user_item_matrix = train_data.pivot_table(index='userId', columns='game', values='rating')
user_item_matrix


'''
game    'Splosion Man  10 Second Ninja X  ...  inFamous: Second Son  page not found
userId                                    ...                                      
1                 NaN                NaN  ...                   NaN             NaN
2                 NaN                NaN  ...                   NaN             NaN
3                 NaN                NaN  ...                   NaN             NaN
6                 NaN                NaN  ...                   NaN             NaN
7                 NaN                NaN  ...                   NaN             NaN
              ...                ...  ...                   ...             ...
7110              NaN                NaN  ...                   NaN             NaN
7116              NaN                NaN  ...                   NaN             NaN
7117              NaN                NaN  ...                   NaN             NaN
7119              NaN                NaN  ...                   NaN             NaN
7120              NaN                NaN  ...                   NaN             NaN

[2831 rows x 2903 columns]
'''

# Fill missing values with 0 [no rating means a rating of 0]
user_item_matrix = user_item_matrix.fillna(0)
user_item_matrix

# Normalize the data
scaler = MinMaxScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)
user_item_matrix_scaled
'''
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
'''
# Calculate cosine similarity
cosine_sim = cosine_similarity(user_item_matrix_scaled, user_item_matrix_scaled)
cosine_sim
'''
array([[1., 0., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.],
       [0., 0., 0., ..., 0., 0., 1.]])
'''

# Convert the similarity matrix into a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
cosine_sim_df

