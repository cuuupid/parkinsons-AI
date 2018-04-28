import pandas as pd 
import numpy as np 
import os, sys

#Extract the features and the predictors
data = pd.read_csv('parkinsons.data')
predictors = data.drop(['name'], axis = 1)
predictors = predictors.drop(['status'], axis = 1).as_matrix()
target = data['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1, 1))
X = scaler.fit_transform(predictors)
Y = target

#Split training data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25, random_state = 7)

#Create the K-Neaarest-Neighbors model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
# make predictions
y_pred = model.predict(X_test)
# summarize the fit of the model
print("k-Nearest Neighbor: ")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))


# Now we are gonna try fine-tuning KNeighborsCclassifier() aka beat 97.959% 