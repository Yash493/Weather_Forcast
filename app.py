from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained models
with open('decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('gradient_boosting_model.pkl', 'rb') as f:
    gradient_boosting_model = pickle.load(f)

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])
        rainfall = float(request.form['rainfall'])

        # Make prediction using the ensemble model
        ensemble_prediction = predict_temperature_ensemble(humidity, pressure, rainfall)

        return render_template('index.html', prediction=ensemble_prediction)
    
    return render_template('index.html')

# Function to make prediction using the ensemble model
def predict_temperature_ensemble(humidity, pressure, rainfall):
    # Create new data for prediction
    new_data = pd.DataFrame({'Humidity': [humidity], 'Pressure': [pressure], 'Rain': [rainfall]})

    # Make prediction using the ensemble of decision_tree_model and gradient_boosting_model
    decision_tree_prediction = decision_tree_model.predict(new_data)
    gradient_boosting_prediction = gradient_boosting_model.predict(new_data)
    ensemble_prediction = (decision_tree_prediction + gradient_boosting_prediction) / 2

    return ensemble_prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
