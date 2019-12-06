# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:55:51 2019

@author: turne
"""
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline


x=pd.read_csv("C:/Users/turne/OneDrive/Desktop/DataFiltered.csv",sep=",",error_bad_lines=False)

x.info()


x.describe()


x.plot(x='population', y='total', style='o')  
plt.title('Population vs Footprint')  
plt.xlabel('Population')  
plt.ylabel('Footprint')  
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(x['population'])

X = x['population'].values.reshape(-1,1)
y = x['total'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

X_train.dropna()

np.savetxt("C:/Users/turne/OneDrive/Documents/cs4347/xtrain.csv", X_train, fmt='%s', delimiter=",")
X_train=pd.read_csv("C:/Users/turne/OneDrive/Documents/cs4347/xtrain.csv",sep=",",error_bad_lines=False)

np.savetxt("C:/Users/turne/OneDrive/Documents/cs4347/ytrain.csv", y_train, fmt='%s', delimiter=",")
y_train=pd.read_csv("C:/Users/turne/OneDrive/Documents/cs4347/ytrain.csv",sep=",",error_bad_lines=False)

np.savetxt("C:/Users/turne/OneDrive/Documents/cs4347/xtest.csv", X_test, fmt='%s', delimiter=",")
X_test=pd.read_csv("C:/Users/turne/OneDrive/Documents/cs4347/xtest.csv",sep=",",error_bad_lines=False)

np.savetxt("C:/Users/turne/OneDrive/Documents/cs4347/ytest.csv", y_test, fmt='%s', delimiter=",")
y_test=pd.read_csv("C:/Users/turne/OneDrive/Documents/cs4347/ytest.csv",sep=",",error_bad_lines=False)




regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

y_pred = regressor.predict(X_test)

y_test.flatten()

y_test.shape

ytest = y_test.values.reshape(1,8636)

ytest.flatten()

y_pred.shape

ypred = y_pred.values.reshape(1,8636)


df = pd.DataFrame({'Actual': ytest, 'Predicted': y_pred},index=[0])
df

predict = {'Actual': ytest, 'Predicted': y_pred}

df = pd.DataFrame(predict, index= [0])

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

