#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
taskData = pd.read_csv("task1_data.csv")
print("Data uploaded successfully")
print(taskData.head(5))
taskData.head(5)

taskData.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

X = taskData.iloc[:, :-1].values
Y = taskData.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

print("Training Complete.")

#plotting the regression line
line = regressor.coef_*X+regressor.intercept_

#plotting for test data
plt.scatter(X,Y)
plt.plot(X,line);
plt.show()

print(X_test)
Y_pred = regressor.predict(X_test)

#Comparing actual vs Predicted

df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(df)

hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No. of Hours = {}".format(hours))
print("Predticted Score = {}".format(own_pred[0]))


from sklearn import metrics
print('Mean Absolute Error : ', metrics.mean_absolute_error(Y_test,Y_pred))