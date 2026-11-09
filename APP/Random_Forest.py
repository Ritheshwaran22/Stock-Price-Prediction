# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:54:07 2022

@author: OKOK D2
"""

#Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read original data and set symbol column as the index
from pathlib import Path
script_dir = Path(__file__).resolve().parent
# Look in APP or parent folder for stock_prices.csv
csv_path = script_dir / 'stock_prices.csv'
if not csv_path.exists():
    csv_path = script_dir.parent / 'stock_prices.csv'
data = pd.read_csv(csv_path, index_col='symbol')
data.columns

#define score tester function to test different models of Random Forest Regressor
def score_tester(model, X_train, X_valid, y_train,y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae_score = mean_absolute_error(y_valid, preds)
    return mae_score

#Split the original dataset into training and testing data
training_data = data.iloc[:246116]
y = training_data.close
test_data = data.iloc[246117:]
#drop 'close' column because that is our target value and drop 'date' column because it is not a continous value to use in predicting
drop_features = ['date', 'close']

#Check which values in training data contain null/NA values
missing_values_in_training_data = training_data.isnull().sum()
print(missing_values_in_training_data)

#Check for NA/null values in testing data 
missing_values_in_test_data = test_data.isnull().sum()
print(missing_values_in_test_data)

#Drop features/columns not needed/helpful for prediction
training_data = training_data.drop(drop_features, axis=1)

#Split training_data into train data for training model and valid data to check the accuracy of the model
X_train, X_valid, y_train, y_valid = train_test_split(training_data,y, train_size = 0.8, test_size=0.2, random_state=0)

#Use SimpleImputer method to fill in all empty/NA values with mean values
my_imputer = SimpleImputer()
imputated_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputated_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputated_X_train.columns = X_train.columns
imputated_X_valid.columns = X_valid.columns

#Creating different RandomForestRegressor models with different parameters. The goal is to see which model is the best to use for prediction.
rfr_model_1 = RandomForestRegressor(n_estimators = 150, random_state = 0)
rfr_model_2 = RandomForestRegressor(max_depth = 15, random_state = 0, criterion='squared_error')
rfr_model_3 = RandomForestRegressor(min_impurity_decrease = 0.6, random_state = 0, criterion='poisson')
rfr_model_4 = RandomForestRegressor(max_features = 'sqrt', random_state = 0)
rfr_model_5 = RandomForestRegressor(n_estimators = 400, random_state = 0)

rfr_models = [rfr_model_1, rfr_model_2, rfr_model_3, rfr_model_4, rfr_model_5]

#Pass each model to the score_tester function with the necessary train and valid data to check the accuracy of each model
count = 1
for model in rfr_models:
    score = score_tester(model, imputated_X_train, imputated_X_valid, y_train, y_valid)
    print("Model ",count,"\t Mean Absolute Error score:", score)
    count +=1
    
#Separate the actual 'close' values we will like to predict from the testing data
actual_values = test_data.close
test_data = test_data.drop(drop_features, axis=1)

#Replace all NA/null values in test data with mean values
my_imputer = SimpleImputer()
imputated_test_data = pd.DataFrame(my_imputer.fit_transform(test_data))

imputated_test_data.columns = test_data.columns

#Predict the 'close' values using the test data with replaced NA values
predictions = rfr_model_2.predict(imputated_test_data)

combined = {'Actual values': actual_values, 'predictions': predictions}
output_df = pd.DataFrame(combined)
output_df.to_csv('output.csv', encoding='utf-8')
print(output_df)

#import pickle library to serialize model object
import pickle

#save the best model to a pickle file
with open('stock_price_model', 'wb') as f:
    pickle.dump(rfr_model_2, f)
    
#Reload the pickle file as a model object
with open('stock_price_model', 'rb') as f:
    mp = pickle.load(f)
    
#Test if model makes same prediction after being serialized
mp.predict(imputated_test_data)