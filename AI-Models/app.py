from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from datetime import datetime, timedelta
import numpy as np

# Load your trained model (ensure rain_model.pkl is in the same directory)
model = pickle.load(open("rain_model.pkl", "rb"))

# Create Flask app
app = Flask(__name__)

# Generate future dates for the next 7 days
future_dates = [datetime.today() + timedelta(days=i) for i in range(7)]
future_dates = [date.strftime("%Y-%m-%d") for date in future_dates]

# Use the correct feature names based on how the model was trained.
# For example, if your model was trained on these columns:
feature_names = ["tavg", "tmin", "tmax", "wdir", "wspd", "pres"]

# Create a DataFrame with dummy (random) future weather data.
# Replace this with actual forecast data if available.
X_future = pd.DataFrame(np.random.rand(7, 6), columns=feature_names)

# Get predictions using the model
predictions = model.predict(X_future)

# Route for homepage: displays a list of future dates and their predicted rainfall.
@app.route("/")
def home():
    return render_template("index.html", predictions=zip(future_dates, predictions))

# Route to predict rainfall for a user-input date from the next 7 days.
@app.route("/predict", methods=["POST"])
def predict():
    try:
        date_input = request.form["date"]
        date_index = future_dates.index(date_input)
        prediction = predictions[date_index]
        return jsonify({"date": date_input, "rainfall_mm": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": "Invalid date or no data available.", "details": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
