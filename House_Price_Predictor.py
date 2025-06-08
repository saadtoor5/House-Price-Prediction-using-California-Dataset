#importing libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Loading the california dataset
housing = fetch_california_housing(as_frame=True)

#Create a dataframe from the dataset
df = housing.frame

df.head()

#Features (independent) and Target (dependent) Variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

#Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Train the linear regression model
model = LinearRegression()
model.fit(X_train,y_train)

#Make Predictions
y_pred = model.predict(X_test)

#Evaluation
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4}\nR2 Score:{r2:.4}\n")

print("Model Coefficents:")
print(f"Intercept: {model.intercept_}\nCoefficents: {model.coef_}")

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficent'])

print("Coefficent for each feature")
print(coef_df)

#Testing the model with new data:

#Function to get the input from the user
def get_user_input():
    print("Enter values for the new house:")
    data = {
        'MedInc': float(input("Median Income (e.g., 4.0): ")),
        'HouseAge': float(input("House Age (e.g., 30): ")),
        'AveRooms': float(input("Average Rooms (e.g., 5.5): ")),
        'AveBedrms': float(input("Average Bedrooms (e.g., 1.0): ")),
        'Population': float(input("Population (e.g., 1200): ")),
        'AveOccup': float(input("Average Occupants (e.g., 3.0): ")),
        'Latitude': float(input("Latitude (e.g., 35.0): ")),
        'Longitude': float(input("Longitude (e.g., -119.0): "))
    }
    return pd.DataFrame([data]) 

# Use safe input values or prompt the user
new_data = get_user_input()

predicted_price = model.predict(new_data)[0]
predicted_price = max(predicted_price, 0)

print(f"\n🏠 Predicted Median House Value: ${predicted_price * 100000:.2f}")