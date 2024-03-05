import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#scikit-learn contains a bunch of useful machine learning libraries
from sklearn import linear_model

#Create a pandas data frame from the CSV file
df = pd.read_csv("Support Files\FuelConsumption.csv")

#Print the top lines of the CSV file we turned into a data frame to confrim its loaded
print(df.head())

#Use the dataframe function 'describe()' to give us a quick desciptive summary of the data
print(df.describe())

#Create a data frame with only the columns we are interested in looking at: Engine size, cylinders, combined fuels consumption, CO2 emissions
modified_df = df[['ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]

#print the top 9 rows of modified dataframe to confirm selection
print(modified_df.head(9))

#Create histograms for the columns we decided upon for our modified data frame
modified_df.hist()
plt.show()

#Create scatter plot graph showing number of cylinders agains the amount of CO2 emisisons produced
plt.scatter(modified_df.CYLINDERS,modified_df.CO2EMISSIONS,color='blue')
plt.xlabel("Number of Cylinders")
plt.ylabel("Amount of CO2 Emissions")
plt.show()

#Create scatter plot graph showing the engine size agains the amount of CO2 emissions
plt.scatter(modified_df.ENGINESIZE,modified_df.CO2EMISSIONS,color='red')
plt.xlabel("Engine Size")
plt.ylabel("Amount of CO2 Emissions")
plt.show()

#We will use the Tain/Test Split approach for our data and our model
#To do this, we will split our data into 80% for training and 20% for testing
#The first thing we need to do is use the numpy.random.rand() function to select a random number of lines from our dataframe up to (80% of the dataframe) and store it in a variable called 'random_rows'
random_rows = np.random.rand(len(df)) < 0.8

#With the random rows now stored, we create create the splits for our dataset. With the training set containing the number of random rows, and the test set containing NOT the random rows used in the training set
    #training_set = df[random_rows] --> Get the 80% of the data
    #test_set = df[~random_rows] --> Get the remaining 20% of the data
training_set = modified_df[random_rows]
test_set = modified_df[~random_rows]

#Create a linear regression model: sklearn.linear_model.LinearRegression
"""
Parameters of sklearn.linear_model.LinearRegression:
fit_intercept: bool, default=True
Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

copy_X: bool, default=True
If True, X will be copied; else, it may be overwritten.

n_jobs: int, default=None
The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

positive: bool, default=False
When set to True, forces the coefficients to be positive. This option is only supported for dense arrays

----------------------------------------------------------------------------------------------

Attributes of sklearn.linear_model.LinearRegression:

coef_: array of shape (n_features, ) or (n_targets, n_features)
Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.

rank_: int
Rank of matrix X. Only available when X is dense.

singular_: array of shape (min(X, y),)
Singular values of X. Only available when X is dense.

intercept_: float or array of shape (n_targets,)
Independent term in the linear model. Set to 0.0 if fit_intercept = False.

n_features_in_: int
Number of features seen during fit.

feature_names_in_: ndarray of shape (n_features_in_,)
Names of features seen during fit. Defined only when X has feature names that are all strings

----------------------------------------------------------------------------------------------

Methods of sklearn.linear_model.LinearRegression:

fit(X, y[, sample_weight]) --> Fit linear model.

get_metadata_routing() --> Get metadata routing of this object.

get_params([deep]) --> Get parameters for this estimator.

predict(X) --> Predict using the linear model.

score(X, y[, sample_weight]) --> Return the coefficient of determination of the prediction.

set_fit_request(*[, sample_weight]) --> Request metadata passed to the fit method.

set_params(**params) --> Set the parameters of this estimator.

set_score_request(*[, sample_weight]) --> Request metadata passed to the score method

"""
regression = linear_model.LinearRegression()

#numpy has a function/routine called asanyarray() that converts a list into a numpy array --> ndarray
#it has 4 parameters:
    #1. a: array_like --> Input data, in any form that can be converted to an array. This includes scalars, lists, lists of tuples, tuples, tuples of tuples, tuples of lists, and ndarrays
    #2. dtype: data-type(optional) --> By default, the data-type is inferred from the input data
    #3. order: {'C', 'F'}(optional) --> Informs whether the array data should be stored in row-major (C-style) or column-major (Fortran-style) order. DON'T FULLY UNDERSTAND THIS!!!!
    #4. like: array_like(optional) --> Reference object to allow the creation of arrays which are not NumPy arrays. If an array-like is passed in supports the __array_function__, the result will be defined by it. In this case, it ensure the creation of an array object compatible with that passed in via this arguement
    
#RETURNS: ndarray or ndarray subclass

#Turn y values into a numpy array
train_x = np.asanyarray(training_set[['ENGINESIZE']])
#Turn y values into a numpy array. Remember y values is what we want to predict
train_y = np.asanyarray(training_set[['CO2EMISSIONS']])

#The 'Fit(X,Y,sample_weight=None)' function is used to fit a linear model to the given data
    # x = {array-like, sparse matrix} of shape (n_samples, n_features) ----> Training data
    # y = {array-like, sparse matrix} of shape (n_samples, n_features) ----> Target values. Will be cast to X's dtype (datatype of x) if necessary
    # sample_weight = {array-like} of shape (n_samples) or (n_samples, n_outputs) ----> Individual weights for each sample. If not provided, all weights will be assumed to be 1.
    
#RETURNS: self: object ---> fitted estimator

#Fit the model with the training data arrays
regression.fit(train_x,train_y)

#As per the comments above, get and print the coofficients and intercept
my_coef = regression.coef_
print("My Coefficients: ",my_coef)
my_intercept = regression.intercept_
print("My Intercept: ",my_intercept)

#Create scatter plot graph showing engine size against the amount of CO2 emisisons produced
plt.scatter(training_set.ENGINESIZE, training_set.CO2EMISSIONS,  color='blue')

#Plot the regression line on top of the scatter plot (!!! NEED TO GOOGLE TO UNDERSTAND WHAT IS GOING ON HERE !!!)
    # ASSUME [0][0] INDICATES STARTING BOTTOM LEFT CORNER AND [0][1] INDICATES ENDING TOP RIGHT CORNER
plt.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()