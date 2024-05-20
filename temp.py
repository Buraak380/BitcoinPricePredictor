from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv('bitcoin_training.csv')

'''
# Printing the dataset
print(datas)
'''

# Train test splitting
datas = datas.drop('Date',axis=1)
print(datas)
X = datas.drop('High',axis=1)
Y = datas['High']

# Splitting the data into training and test sets

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)

# Scaling the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# Training the model. I used RandomForestRegressor in this one.

rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

print('The accuracy for this model is : % {}'.format(round(r2*100,2)))


# Data visualizatio
# Actual and predicted values for the training set
plt.figure(figsize=(10, 6))
plt.scatter(y_train, rf_regressor.predict(X_train), color='blue', label='Actual')
plt.scatter(y_train, y_train, color='red', label='Prediction')
plt.title('Training Set - Actual and Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Actual and predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Data Points')
plt.plot(y_test, y_test, color='red', label='True Prediction', linestyle='--')
plt.title('Test  - Actual and Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()












