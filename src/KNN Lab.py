from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Import the load_dotenv to allow us to use environment variables from the .env file of the project
from dotenv import load_dotenv
import os

#load the .env variables
load_dotenv()

#use os.environ or os.getenv to access environment variables as if they are from ACTUAL enviornment and not .env file
print(os.getenv("KNN_DATASET_URL"))

# import data

# Import from URL
#df = pd.read_csv(os.getenv("KNN_DATASET_URL"))

#Use previously downloaded data file
                #Use double slashes to escape the special characters being detected
df = pd.read_csv("Machine Learning with Python\\Support Files\\teleCust1000t.csv")
df.head()

print(df.head())

#Get a count of the different labels rom  the 'custcat' column
print("Count of unique labels in 'custcat' field: ",df['custcat'].value_counts())
"""RESULTS:
    3    281
    1    266
    4    236
    2    217
    Name: custcat, dtype: int64"""

#Print columns of dataframe/dataset
print("Dataframe Colums: ",df.columns)

# Get descriptive statistics of income column
print(df['income'].describe())

# Display histogram to show the data distribution - column is the column to be plotted, bins is the size of the interval/width of the bars in the histogram
df.hist(column='income', grid=True,bins=50)
plt.show()
"""FINDINGS:
    The histogram showed skewed RIGHT distributiuon of income amounts
"""

# To use the Scikit-learn library, we need to convert the Pandas dataframe to a Numpy array
X_values = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

#Print the first 5 rows to confirm the conversion and ensure the data is in the correct format
print(X_values[0:5])

# Get our datset's Y values (Target values)
Y_values = df['custcat'].values

#Print the first 5 rows to confirm the conversion and ensure the data is in the correct format
print(Y_values[0:5])

# Normalise Data
# Standardise features/columns by removing the mean and scaling to unit variance
#Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:

"""
----------------------------------------------------------------------------------
NEED TO SEARCH AND INCLUDE WHAT IS ACTUALLY GOING ON IN THIS FUNCTION CALL TO UNDERSTAND HOW IT WORKS AND WHY THE VALUES CHANGE THE WAY THEY DO
----------------------------------------------------------------------------------
"""

X_values = preprocessing.StandardScaler().fit(X_values).transform(X_values.astype(float))

#Display the first 5 rows to confirm the conversion and ensure the data is in the correct format
print(X_values[0:5])