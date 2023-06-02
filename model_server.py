from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load the dataset
data = pd.read_csv('weather_data_final.csv')

# Split the dataset into input features and target variable
X = data[['Humidity', 'Pressure', 'Rain']]
y = data['Temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Regressor model
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_regressor.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error (MSE):', mse)
print('R-squared Score (R2):', r2)


# Assuming you have already trained and fitted the Decision Tree Regressor model

# Prepare new data for prediction
new_data = pd.DataFrame({'Humidity': [60], 'Pressure': [1013], 'Rain': [0.2]})

# Make prediction
prediction = dt_regressor.predict(new_data)

# Print the predicted temperature
print('Predicted temperature:', prediction[0])



# Save the Decision Tree Regressor model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dt_regressor, file)