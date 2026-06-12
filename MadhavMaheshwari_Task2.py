import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"c:\Users\kmahe\Downloads\Housing.csv")
print(df.head())
print(df.info())

print(df.isnull().sum())       # Check for missing values
print(df.duplicated().sum())   # Check for duplicate values
df.dropna(inplace=True)        # Drop rows with missing values
print(df.describe())           # Get summary statistics

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')        # Visualize the correlation matrix
plt.title('Correlation Matrix')
plt.show()

X = df[['area' , 'bedrooms', 'bathrooms' , 'stories', 'parking']]        # Features
y = df['price']        # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        # Split the data into training and testing sets
model = LinearRegression()        # Create a linear regression model
model.fit(X_train, y_train)        # Train the model

y_pred = model.predict(X_test)        # Make predictions on the test set

mse = mean_squared_error(y_test, y_pred)        # Calculate mean squared error
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)        # Calculate R-squared score
print("R-squared Score:", r2)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)        # Visualize actual vs predicted values
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

plt.scatter(X_test['area'], y_test, color='blue', label='Actual Prices')        # Visualize the relationship between area and actual prices
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted Prices')        # Visualize the relationship between area and predicted prices    
plt.title('Area vs Prices')
plt.xlabel('Area')  
plt.ylabel('Prices')
plt.legend()    
plt.show()