# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:17:30 2021

@author: ASUS
"""

import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle


dataframe = pandas.read_csv("forestfires.csv")
print(dataframe)

seed = 7
numpy.random.seed(seed)


dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

print(dataframe.head())

print("Statistical Description:", dataframe.describe())

print("Shape:", dataframe.shape)

print("Data Types:", dataframe.dtypes)

print("Correlation:", dataframe.corr(method='pearson'))

plt.hist((dataframe.area))

dataframe.hist()


dataframe.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)


num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVM', SVR()))

# Evaluations
results = []
names = []
scoring = []

dataset = dataframe.values
X = dataset[:,0:12]
print(X)
Y = dataset[:,12]
print(Y)

for name, model in models:
    # Fit the model
    model.fit(X, Y)
    
    predictions = model.predict(X)
    
    # Evaluate the model
    score = explained_variance_score(Y, predictions)
    mae = mean_absolute_error(predictions, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mae)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, score, mae)
    print(msg)

#'temp' has the highest correlation with the area of forest fire(which is a positive correlation), followed by 'RH' also a positive correlation, 
#'Rain' has the least correlation




#Feature Selection
model = ExtraTreesRegressor()
print(model)
rfe = RFE(model, 5)
print(rfe)
#fitting model on train data
fit = rfe.fit(X,Y)
print(fit)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 
print(dataframe.columns)
df=dataframe[['DMC','DC','temp','RH','wind','area']]
print(df)

x=df.iloc[:,0:5]
print(x)
y=df.iloc[:,-1]
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model.fit(x_train,y_train)
score = model.score(x_train, y_train)
print("Score: ", score)

cv_scores = cross_val_score(model, x_train,y_train,cv=10)
print("Mean cross-validataion score: %.2f" % cv_scores.mean())
#predict the test data by using the trained model. After the prediction, we'll check the accuracy level by using the MSE and RMSE metrics.

y_predict = model.predict(x_test)
print(y_predict)

mse = mean_squared_error(y_test, y_predict)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % mse**(0.5))



pickle.dump(model, open('etrmodel.pkl','wb'))




x_ax = range(len(y_test))
plt.plot(x_ax, y_test, lw=0.6, color="blue", label="original")
plt.plot(x_ax, y_predict, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
