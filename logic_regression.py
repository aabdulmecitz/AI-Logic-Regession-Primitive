import numpy as np
import pandas as pd

#definition a method
def dummy(parameter):
  dummy_parameter = parameter + 5
  return dummy_parameter;


# initilaize params
# np.full() function takes array size for first arguement. But we can enter 2 dimension array for example x, 1
# in this situation the function creates x,1 matrix, then the function fills matrix with value of second arguement.
def init_wight_and_bias(dimension):
  w = np.full((dimension, 1), 0.1)
  b = 0;
  return w, b

#z = np.dot(w.T , x_train) + b
def sigmoid(z):
  y_head = 1/(1+np.exp(-z))
  return y_head
#y_head = sigmoid(z)

# if we want to multiply two matrix, first one's column size equals second one's rows size
def forward_propagation(w, b, x_train, y_train):
  z = np.dot(w.T, x_train) + b
  y_head = sigmoid(z) # probabilitisctic value
  loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
  cost = (np.sum(loss))/x_train.shape[1]
  return cost

def forward_backward_propagation(w, b, x_train, y_train):

  #forward propagation
  z = np.dot(w.T, x_train) + b
  y_head = sigmoid(z) # probabilitisctic value
  loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
  cost = (np.sum(loss))/x_train.shape[1]

  #backward propagation
  derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
  derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
  gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
  return cost, gradients

def update(w, b, x_train, y_train, learning_rate, num_of_iteration):
  cost_list = []
  cost_list2 = []
  index = []
  for i in range(num_of_iteration):
    cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
    cost_list.append(cost)
    w = w - learning_rate * gradients["derivative_weight"]
    b = b - learning_rate * gradients["derivative_bias"]
    if i % 10 == 0:
      cost_list2.append(cost)
      index.append(i)
      print("Cost after itteration %i: %f" %(i, cost))

#prediction

def predict(w, b, x_test):
  z = sigmoid(np.dot(w.T, x_test) + b)
  Y_prediction = np.zeros((1, x_test.shape[1]))

  for i in range(z.shape[1]):
    if z[0, i] <= 0.5 :
      Y_prediction[0, i] = 0
    else:
      Y_prediction[0, i] = 1
  return Y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learing_rate, num_iterations):
  dimention = x_train.shape[0]
  w, b = init_wight_and_bias(dimention)
  parameters, gradients, cost_list = update(w, b, x_train, y_train, learing_rate, num_iterations)

  Y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
  Y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

  print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train))*100))
  print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test))*100))


logistic_regression(x_train, y_train, x_test, y_test, learing_rate=0.01, num_iterations=150)
