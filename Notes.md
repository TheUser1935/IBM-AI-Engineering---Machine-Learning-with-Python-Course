# Machine Learning with Python Course Notes

This file contains the notes I have taken during the course. I decided to add my notes into this repository as an aid for other learners that might be interested. Original notes were done in my personal Notion.

# Supervised vs Unsupervised Learning

**Supervised Learning**

- _Classification_: classifies labelled data
- _Regression_: Predicts trends using previous labelled data
- Has more evaluations methods than unsupervised learning
- Controlled environment

**Unsupervised Learning**

- _Clustering_: Finds patterns and groupings from unlabelled data
- Has fewer evaluation methods than sueprvised learning
- Less controlled Environment

# Intro to Regresison

## What is regression?

- process of predicting a continuous value
- Looks at independent variables (explanatory or causal) and how they impact the final state/goal/metric we try and predict
  ![Overview](<Lesson Notes Images/Regression/overview.png>)

- The dependent variable - or goal we try and predict has to be continuous, however, the independent variables can be continuous or categorical values

## Types of regression models

- Simple Regression - 1 independent variable used to predict dependent variable. e.g. predict CO2 emission vs Engine size of cars
  - Simple linear regression
  - Simple non-linear regression
- Multiple regression - use of multiple independent variables to predict dependent variable e.g. predict CO2 emission2 vs engine size AND cylinders of cars
  - multiple linear regression
  - multiple non-linear regression

## Applications of regression

- Sales forecasting - predict individuals sales based upon age, education, years of experience
- Satisfaction analysis - Psychology - satisfaction based upon demographic and psychological factors
- House prices - size, bedrooms, etc
- employment income - hours worked, age, sex, field, experience, etc
- Can really be applied across very many domains

## Regression algorithms

- ordinal regression
- poisson regression
- fast forest quantile regression
- linear, polynomial, Lasso, stepwise, ridge regression
- bayesian linear regression
- neural network regression
- decision forest regression
- boosted decision tree regression
- KNN (K-nearest neighbours)

# Simple Linear Regression

- We can look at independent variables and identify one that we want to see if there is a linear relationship to the dependent variable we want to be able to predict
- For instance, we can plot the engine size of cars against the amount of CO2 emissions to se eif there is a linear relationship
  ![Simple Linear Regression Overview](<Lesson Notes Images/Regression/simple_Linear_Regression_Overview.png>)

- In the image above, we can see that as the engine size increases, there is an increase in the amount of CO2 emissions and we can draw a line through our data.
- If the line we draw is accurate and a good measure of our data, we can use it to build a model to predict the amount of CO2 based upon an engine size
- The fit line is traditionally shown as a polynomial
  - $\widehat{Y} = \theta_0 + \theta_1 X_1$
  - $\theta$ = theta
  - $\theta_0$ and $\theta_1$ are the parameters of the line we must adjust. They are the coefficients of the linear equation
  - $\theta_0$ = known as the intercept
  - $\theta_1$ = slope or gradient of the fitting line
  - $\widehat Y$ = response variable/ dependent variable
  - $X_1$ = independent variable

## Simple Linear Regression Equations

2 approaches to adjusting our params to find best fitting line:

- mathematical
- optimisation
- Mathematical - Calculate theta 0 and theta 1 using their prescribed algorithms

### Theta zero and Theta 1

- Mathematical approach to adjusting our params

  $$
  \large \theta_1 =\space \frac{\sum^s_{i=1}(x_i-\=x)(y_i - \=y)}{\sum^s_{i=1}(x_i-\=x)^2}
  $$

  - At first, this equation looks psycho but isn’t so bad when walked through slowly and broken down:
    - $\large \sum^s_{i=1}$ = We want the total sum of the following equation for essentially each row of x and y, indexed as i
    - $\large (x_i-\=x)$ = Current value of x, minus the AVERAGE of x —> $\large \=x$ = average of x
    - $\large (y_i-\=y)$ = current value of y (our dependent value) minus the AVERAGE of y —> $\large \=y$ = average of y
    - Using this information of understanding the equation, we can also make sense of the bottom half of the equation
      $$
      \large \theta_0 = \=y -\theta_1 \=x
      $$
  - $\large \theta_0$ (theta 0) equation is much easier to understand once you have grasped the $\large \theta_1$ equation - This equation asks us for the AVERAGE of the y values, MINUS the $\large \theta_1$ value from its equation, MULTIPLIED by the AVERAGE value of x values
    ![Estimate Params](<Lesson Notes Images/Regression/theta_estimate_params.png>)

  - We can then make a prediction of our dependent variable based upon the independent variable and use the equations shown above
    ![Prediction with linear regression](<Lesson Notes Images/Regression/predict_with_params_simple.png>)

## Pros of linear regression

- Very fast
- not tuning of parameters
- easy to understand and highly interpretable

# Model Evaluation in Regression Models

## Model evaluation approaches

- Course will discuss 2 types of evaluation approaches:
  - Train and Test on the same dataset
  - Train/Test Split
- Train and Test on Same Dataset:
  - Complete raining of our model on our data set
  - then we take a portion of that dataset, removing the actual dependent variable value and keeping the independent variable/s
  - With those independent variables, we then pass that into our model to carry out the predictions and compare the result to the ACTUAL values of the dependent variable
  - This will indicate how accurate our model actually is
    ![Model Evaluation](<Lesson Notes Images/Regression/simple_lin_model_eval.png>)
  ### Considerations of train and test on the same data set
  - Will have High “training accuracy”(on the dataset it was trained on)
  - Most likely have Low “out-of-sample-accuracy” (data that was not in the dataset that it was trained on)
  ## Calculating the accuracy of a model
  - There are a number of metrics to use to indicate accuracy, one of the simplest is comparing the values of model and actual values, as discussed above
  - We can call this the _Error_ value
  - $\large Error/Mean\space Absolute\space Error(MAE)=\frac{1}{n} \displaystyle\sum_{j=1}^n | y_j - \widehat y_j |$
    - $\large y$ = Actual value of dependent variable
    - $\large \widehat y$ = Predicted value of dependent variable
    - All this equation really says is that for each row of data we want to SUBTRACT the PREDICTED y value, FROM the ACTUAL y value and record the SUM
      - Then DIVIDE that sum by the NUMBER OF ROWS OF DATA used —> to give us the average metric of ERROR
    - Essentially, it can be summarised as:
      - calculate the average difference between predicted value and actual value from the sets of data tested

## What is training and out-of-sample accuracy?

- Training accuracy:
  - % of correct predictions when using the training data set
  - Not always a good thing, possible to have result of ‘over-fitting’
    - Over-fit - the model is overtrained to the dataset, which may capture noise and produce a non-generalised model
- Out-of-Sample accuracy
  - % of correct predictions that the model makes on data the model has not been trained on
  - Train and test on same dataset tends to have low accuracy on out-of-sample data due to overfit
  - It’s important that our models have a high, out-of-sample accuracy because our models aim is to make correct predictions on new datasets

## Train/Test split evaluation approach

- This approach sees us split the dataset we have into two sections:
  - A training set
  - Testing set
- This allows us to train our model on the data we have but at the same time reserving some of the data to be tested on as out-of-sample
- This allows us to carry out accuracy tests, like the error metric discussed before
  ![Train/Test Split](<Lesson Notes Images/Regression/simp_lin_test_split.png>)

![Train/Test Evaluation Approach](<Lesson Notes Images/Regression/simp_lin_test_split_eval.png>)

## K-Fold Cross-Validation and how to use it

- Set aside a percentage of the dataset to use for testing (e.g. first 25%), use the rest for training, then use the next group of percentage for testing and use the rest for training, keep repeating
  - Record the accuracy for each grouped testing, which can be used to calculate the average score
    ![K-Fold Approach](<Lesson Notes Images/Regression/k-cross-fold-approach.png>)
- Each fold is unique, where no training data used in one fold is used in another
- This method is essentially multiple rounds of Train/Test Split using the same dataset where each split is different

# Evaluation Metrics in Regression Models

- Will review the following metrics of evaluation:
  - MAE - Mean Absolute Error
  - MSE - Mean Squared Error
  - RMSE - Root Mean Squared Error
  - RAE - Relative Absolute Error
  - RSE - Relative Squared Error

## What is an error of the model?

- Error: measure of how far the data is from the fitted regression(trend) line
  ![What is Error of Model](<Lesson Notes Images/Regression/simpl_lin_error_model.png>)

- $\large Error/Mean\space Absolute\space Error(MAE)=\frac{1}{n} \displaystyle\sum_{j=1}^n | y_j - \widehat y_j |$
- $\large Mean\space Squared\space Error(MSE)=\frac{1}{n} \displaystyle\sum_{j=1}^n ( y_j - \widehat y_j )^2$
  - The MSE is often more popular than the MAE because it focusses more on LARGE errors as it exponentially increases large errors in comparison to small ones
  - The mean of all residual errors
- $\large Root \space Mean\space Squared\space Error(MSE)=\sqrt {\frac{1}{n} \displaystyle\sum_{j=1}^n ( y_j - \widehat y_j )^2}$
  - This is just the squared root of the MSE
  - It is one of the most popular metrics because it is interpretable as the same units as the $\large y$ units, making it easy to relate its information
- $\large Relative \space Absolute \space Error(RAE)= \huge \frac{\sum^n_{j=1}|y_j-\widehat y_j|}{\sum^n_{j=1}|y_j-\bar y_j|}$
  - $\large \bar y$ = mean value of y (dependent)
  - Takes the total absolute error and normalises it by dividing it by the total absolute error of the simple predictor
- $\large Relative \space Sqaured\space Error(RSE)= \huge \frac{\sum^n_{j=1}(y_j-\widehat y_j)^2}{\sum^n_{j=1}(y_j-\bar y_j)^2}$
  - $\large \bar y$ = mean value of y (dependent)
  - Very similar to RAE but is widely adopted by the data science community as it is used for calculating $R^2$
- $\large R^2 = 1-RSE$
  - $R^2$ is not en error per say, but is a popular metric for the accuracy of your model
  - It represents how close the data points are to the fitted regression line
  - The higher the value of $R^2$, the better the model fits your data

# Multiple Linear Regression

## Brief Overview

- Multiple Linear regression is having multiple independent variables (x values) impact the dependent variable (y value)
- It is an extension of the simple linear regression
- EXAMPLE: predict CO2 Emissions vs (Engine Size and Num of Cylinders)
- Still have the limitations of dependent variable being a CONTINUOUS value

## Applications of Multiple Linear Regression

- There are two main applications:
  - Identify strength of the effect the independent variables have on the prediction dependent variable
    - EXAMPLE: Does revision time, test anxiety, lecture attendance and gender have any effect on the exam performance of students?
  - Understand how the DEPENDENT variable changes when we change the independent variable
    - EXAMPLE: How much does blood pressure go up or down for every unit increase or decrease in the BMI of a patient?

## Multiple Linear Regression Equations

- Will base the work off the car data we have
- EQUATION FOR MULTIPLE LINEAR REGRESSION
  - $\huge \widehat y=\theta_0+\theta_1x_1+\theta_2x_2+\space ...\theta_nx_n$
    - $\widehat y$ = Predicted y value
    - $\theta_0$ = the intercept
    - $\theta_1$ = theta value for feature set column 1
    - $x_1$ = feature set column 1 value
  - We can also show it as a vector form:
    - $\huge \widehat y=0^TX$
    - $\large \theta ^T$ = Theta transposed
      ![Multiple Regression Theta Transposed](<Lesson Notes Images/Regression/multiple_regr_theta_transposed.png>)
    - X = the feature set. Being the components of the data. E.g X1 = engine size, X2 = num of cylinders, etc.
      - The first element of the feature set is set to 1 because it turns the theta zero into the intercept or bias parameter
        ![Multiple Regression X values Vector](<Lesson Notes Images/Regression/multiple_regr_x_values.png>)
  - When $\large \theta ^TX$ is in a 1 dimensional space it is an equation of a line (like simple linear regression)
    - In higher dimension, where we have more than 1 input, the line is called a plane or a hyper plane and this is what is used for multiple linear regression

## Best fit plane/hyper plane Goal

- The goal is to find the best fitting plane for our data
- To do this we should estimate the values of theta vector that best predict the value of the target field (y) in each row
- To achieve this, we need to minimise the error of the prediction
-

## optimised parameters for theta vector

- Firs thing is to find what the optimised parameters are (e.g engine size, cylinders, number of seats), then we will find a way to optimise the parameters
- OPTIMISED PARAMETERS ARE PARAMETERS THAT LEAD TO THE FEWEST ERRORS

## Using MSE to find the errors in the model - continuation of optimised params

- We will work through using MSE to calculate errors, with the assumption we have already solved $\theta$ (theta) vector values
- We can then use the feature sets (x values) to predict the dependent variable (y value) for a car
  ![Multiple Regression Demo Data](<Lesson Notes Images/Regression/mult_lin_regr_start_data.png>)
- If we plug the feature sets values into our equation we find $\large \widehat y$ (predicted value)
  ![Multiple Regression Demo Y Predicted](<Lesson Notes Images/Regression/mult_lin_regr_demo_y_pred.png>)
- We can then compare the ACTUAL value of $\large y_i$
  ![Multiple Regression Demo Y Actual](<Lesson Notes Images/Regression/mult_lin_regr_demo_y_actual.png>)
- We then find how DIFFERENT the predicted value is from the actual value (**residual error**)
  - $\large y_i -\widehat y_i$
    ![Multiple Regression Demo Actual vs Predicted](<Lesson Notes Images/Regression/mult_lin_regr_demo_y_actual_vs_pred.png>)
- The mean of all residual errors (for each row of x and y values) can show how bad the model is representing the data set - This is called the Mean Squared Error → which was discussed briefly earlier
- $\large Mean\space Squared\space Error(MSE)=\frac{1}{n} \displaystyle\sum_{j=1}^n ( y_j - \widehat y_j )^2$

## How do we find the parameter or coefficients of multiple linear regression?

### How to estimate $\large \theta$ (theta) vector?

- Few approaches, but the most common methods are:
  - Optimisation algorithm approach
  - Ordinary Least Squares

### Ordinary Least Squares

- Tries to estimate the values of the coefficients by minimising the MSE (mean squared error)
- uses the data as a matrix and uses Linear algebra operations to estimate the optimal values for the theta
- CONSIDERATIONS:
  - It can take a long time to complete this operations over large datasets (10K+ rows) due to the time complexities

### Optimisation algorithm

- Use a process iteratively minimising the error of your model on the training set
- An example of this is **Gradient Decent**
  - starts by using random values for the coefficients and then calculates the errors and tries to minimise it through y’s changing of the coefficients in multiple iterations
  - This is the proper approach if you have a large data set

## Making Predictions with multiple linear regression after finding optimised parameters

![Multiple Regression Making Predictions](<Lesson Notes Images/Regression/mult_lin_regr_pred_post_optim_params.png>)

- The values identified for each $\large \theta_i$ (e.g. theta 0, theta 1, etc, etc) indicate the predicted impact of that variable on the dependent variable
  - So looking at the example above, it shows that the number of cylinders has a higher impact on the prediction compared to the engine size

## Q&A on multiple linear regression

- How to determine whether to use simple or multiple linear regression?
  - sometimes multiple independent variables results in a better model than a simple linear model
- How many independent variables should you use?
  - Adding too many unjustified independent variables can result in an overfit model
- Should independent variables be continuous?
  - categorical independent variables can be incorporated into regression models by converting them into numerical values
    - e.g. TRANSMISSION TYPES
      - manual = 0
      - automatic = 1
- What are the linear relationships between the dependent variable and the independent variables
  - Remember that these approaches require a linear relationship
  - visualising the data with scatter plots can indicate a linear relationship, if there is no linear pattern then it is likely not suitable for linear regression models and should instead use a NON-LINEAR regression model

## Bonus - Explained Variance Regression Score

- Let $\large \hat y$ be the estimated target output, y the corresponding (correct) target output, and Var be the Variance (the square of the standard deviation). Then the explained variance is estimated as follows:
- $\large \texttt{ExplainedVariance}(y, \hat{y} = 1 - \frac{Var\{y-\hat{y}\}}{Var\{y\}}$
- The best possible score is 1.0, the lower values are worse.

---

# K-Nearest Neighbours (Classification)
