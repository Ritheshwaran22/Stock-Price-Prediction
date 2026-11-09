#Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

#Upload the data
#Getting the data from GitHub repo with raw link of .csv file.

url = 'https://raw.githubusercontent.com/egeoguzman/svm_tesla_stock_prediction/master/TSLA.csv'
df = pd.read_csv(url)
df

#X & Y data sets as a lists

dates = []
prices = []

#Will try to predict last 5
df.tail(1)

#This dataframe shows rest of the list
df = df.head(len(df)-1)
df

#Gets the columns

df_dates = df.loc[:,'Date']
df_open_price = df.loc[:,'Open']

#Pars the data and sets as a independent data set X

for date in df_dates:
  dates.append( [int(date.split('-')[2])])

#Pars the data and sets as a independent data set Y

for open_price in df_open_price:
  prices.append(float(open_price))

print(dates)
print(prices)

def predict_prices(dates,prices,x):

  #Defines the SVM regression models.


  svr_lin = SVR(kernel='linear' , C=1e3)
  svr_poly = SVR(kernel='poly' , C=1e3 , degree=2)
  svr_rbf = SVR(kernel='rbf' , C=1e3 , gamma=0.1)



  #Train the models.
  svr_lin.fit(dates,prices)
  svr_poly.fit(dates,prices)
  svr_rbf.fit(dates,prices)


  #Linear Regression Model and train it.

  lin_reg = LinearRegression()
  lin_reg.fit(dates,prices)

  # Data Visualization

  plt.scatter(dates, prices, color='black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR RBF')
  plt.plot(dates, svr_poly.predict(dates), color='blue', label='SVR Poly')
  plt.plot(dates, svr_lin.predict(dates), color='green', label='SVR Linear')
  plt.plot(dates, lin_reg.predict(dates), color='orange', label='Linear Reg')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.title('Regression')
  plt.legend()
  plt.show()

  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0],svr_poly.predict(x)[0],lin_reg.predict(x)[0]

predicted_price = predict_prices(dates,prices,[[6]])
print(predicted_price)