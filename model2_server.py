from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load data
data = pd.read_csv('weather_data_final.csv')

# Split data into input features (X) and target variable (y)
X = data[['Humidity', 'Pressure', 'Rain']]
y = data['Temperature']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create gradient boosting model
clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)

# Evaluate model performance
train_r2 = clf.score(X_train, y_train)
test_r2 = clf.score(X_test, y_test)
print('Training R-squared:', train_r2)
print('Testing R-squared:', test_r2)

# Make a prediction on new data
new_data = pd.DataFrame({'Humidity': [60], 'Pressure': [1013], 'Rain': [0.2]})
prediction = clf.predict(new_data)
print('Predicted temperature:', prediction[0])

# Save the Gradient Boosting Regressor model
with open('gradient_boosting_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
