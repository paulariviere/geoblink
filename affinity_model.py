#!/usr/bin/env python
import psycopg2 as pg
import sys, os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import pipeline
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt

# Connection to the database. This is for the Postgres server located at localhost
connection = pg.connect("dbname=geoblink user=postgres password=postgres")

# Import the feature matrix X and the affinities (target) y from the database
X = psql.read_sql_query("SELECT * FROM X", connection)
y = psql.read_sql_query("SELECT * FROM y", connection)

# First, calculate p-scores to determine which features are not needed
# (i.e. which of them have a probability of the null-hypothesis, p-value, higher than 0.05)
print("******************** P-SCORES *********************")
print("Calculating p-scores")
F, pvalues = f_regression(X, y.values.ravel(), center=True)
print("p-scores higher than 0.05:")
print(pvalues[pvalues>0.05])
indices = np.nonzero(pvalues > 0.05)
print("Features with high p-score:")
print(X.columns.values[indices])

# Split the matrices into train and test sets:
x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.11,random_state=10)

print("************ LINEAR REGRESSION MODEL *************")
# Calculate the least squares regression for the data (training set)
lm = linear_model.LinearRegression()
model = lm.fit(x_train,y_train)

# Predictions for the affinities
print("Prediction:")
predictions = lm.predict(x_train)
print(predictions)[0:5]

# Accuracy of the method
print("Score for the train set:")
print(lm.score(x_train,y_train))
print("Score for the test set:")
print(lm.score(x_test,y_test))

print("************* RIDGE REGRESSION MODEL *************")
# Calculate the results for different regularization coefficients
n_alphas = 100
alphas = np.logspace(-6, -4, n_alphas)

# Initialize arrays
errors = []
score_train = []
score_test = []

# Initialize optimal values
alpha_optimal = 0
score_test_optimal = 0.
score_train_optimal = 0.

# Loop in alpha values
for a in alphas:
    # Run the Ridge model
    rm = pipeline.make_pipeline(linear_model.Ridge(alpha=a))
    rm = linear_model.Ridge(alpha=a)
    rm.fit(x_train,y_train)

    # Calculate the mean squared error and add it to the array
    mse = metrics.mean_squared_error(y_test, rm.predict(x_test))
    errors.append(np.sqrt(mse))    

    # Calculate the score for the training set
    current_score_train = rm.score(x_train,y_train)
    score_train.append(current_score_train)

    # Calculate the score for the test set
    current_score_test = rm.score(x_test,y_test)
    score_test.append(current_score_test)

    # Print results
    # Remove comment to see results:
    ### print(a,np.sqrt(mse),rm.score(x_train,y_train),rm.score(x_test,y_test))

    # Update the optimal values for alpha and for the scores
    if (current_score_test > score_test_optimal):
      score_test_optimal = current_score_test
      score_train_optimal = current_score_train
      alpha_optimal = a

# Print errors
print(errors)

# Print optimal values
print("Optimal value of alpha: " + str(alpha_optimal))
print("Score for the train set:")
print(str(score_train_optimal))
print("Score for the test set:")
print(str(score_test_optimal))

# We can plot the results of the scores as a function of alpha:
#plt.plot(alphas,score_train)
#plt.plot(alphas,score_test)
#plt.show()
