# Algorithms-of-Machine-Learning
Implementation of some useful algorithms of machine learning using Octave programming language, an open-source alternative to MATLAB.

## Linear Regression

- ***Linear regression with one variable:***
<br /> In this part, we implement linear regression with one
variable to predict profits for a food truck. Suppose you are the CEO of a
restaurant franchise and are considering different cities for opening a new
outlet. The chain already has trucks in various cities and you have data for
profits and populations from the cities.
We would like to use this data to help you select which city to expand to next.
The file data1.txt contains the dataset for our linear regression problem.
The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a loss.

- ***Linear regression with multiple variables:***
<br /> In this part, we implement linear regression with multiple variables to
predict the prices of houses. Suppose you are selling your house and you
want to know what a good market price would be. One way to do this is to
first collect information on recent houses sold and make a model of housing prices.
The file data2.txt contains a training set of housing prices in Portland, Oregon.
The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

- ***Regularized Linear Regression and Bias-Variance Tradeoff:***
<br /> In this part, we implement regularized linear regression to predict the amount of
water flowing out of a dam using the change of water level in a reservoir. Then we use this
algorithm to study models with different bias-variance properties.

## Logistic Regression

- ***Logistic Regression (without regularization):***
<br /> In this part, we build a logistic regression model to
predict whether a student gets admitted into a university.
Suppose that you are the administrator of a university department and
you want to determine each applicant’s chance of admission based on their
results on two exams. You have historical data from previous applicants
that you can use as a training set for logistic regression. For each training
example, you have the applicant’s scores on two exams and the admissions decision.
We build a classification model that estimates an applicant’s
probability of admission based the scores from those two exams.

- ***Regularized Logistic Regression:***
<br /> In this part, we implement regularized logistic regression to predict
whether microchips from a fabrication plant passes quality assurance (QA).
During QA, each microchip goes through various tests to ensure it is functioning correctly.
<br /> Suppose you are the product manager of the factory and you have the
test results for some microchips on two different tests. From these two tests,
you would like to determine whether the microchips should be accepted or
rejected. To help you make the decision, you have a dataset of test results
on past microchips, from which you can build a logistic regression model.

- ***One-vs-all Classification:***
<br /> In this part, We extend our previous implemention of logistic regression
and apply it to one-vs-all classification and use it to recognize handwritten
digits (from 0 to 9).
We implement one-vs-all classification by training multiple regularized logistic
regression classifiers, one for each of the K classes in our dataset. In the handwritten digits dataset,
K = 10, but our code works for any value of K.

## Neural Network

Implementation of feedforward propagation and the backpropagation algorithm for neural networks and use it to predict handwritten digits.

