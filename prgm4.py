                                                                                                                                                                                                       # Import necessary libraries
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Convert to a DataFrame for easier manipulation
df = pd.DataFrame(X, columns=boston.feature_names)
df['PRICE'] = y

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error: {:.2f}".format(mse))
print("R^2 Score: {:.2f}".format(r2))

# Example prediction
example_input = X_test[0].reshape(1, -1)
predicted_price = model.predict(example_input)
print("\nExample Prediction:")
print("Predicted price: ${:.2f}".format(predicted_price[0]))