import os, sys, shutil, time

from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from geopy.geocoders import Nominatim

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/images/<filename>')
def download_file(filename):
    image_directory = os.path.join(app.root_path, 'static', 'images')
    return send_from_directory(image_directory, filename)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/work.html')
def work():
    return render_template('work.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/result.html', methods=['POST'])
def predict():
    rfc = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'rf_model'))
    print('Model loaded')

    if request.method == 'POST':
        address = request.form['Location']
        timestamp_str = request.form['timestamp']

        # Geocoding
        geolocator = Nominatim(user_agent="crime_predictor_app_v1")
        location = geolocator.geocode(address, timeout=10)

        if location is None:
            return render_template('result.html', prediction='Error: Location not found. Please enter a more specific address.')

        print(location.address)
        lat = [location.latitude]
        lon = [location.longitude]
        latlong = pd.DataFrame({'latitude': lat, 'longitude': lon})
        latlong['timestamp'] = timestamp_str

        # Move timestamp to first column
        cols = latlong.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = latlong[cols]

        # Parse datetime from YYYY-MM-DDTHH:MM format (from datetime-local input)
        # Pandas should infer this format correctly without a format string.
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        if data['timestamp'].isna().any():
            return render_template('result.html', prediction='Error: Invalid date/time format. Please ensure you select a date and time.')

        column_1 = data.iloc[:, 0]
        DT = pd.DataFrame({
            "year": column_1.dt.year,
            "month": column_1.dt.month,
            "day": column_1.dt.day,
            "hour": column_1.dt.hour,
            "dayofyear": column_1.dt.dayofyear,
            "week": column_1.dt.isocalendar().week,
            "weekofyear": column_1.dt.isocalendar().week,
            "dayofweek": column_1.dt.dayofweek,
            "weekday": column_1.dt.weekday,
            "quarter": column_1.dt.quarter,
        })

        data = data.drop('timestamp', axis=1)
        final = pd.concat([DT, data], axis=1)

        # Check for missing values
        if final.isna().any().any():
            return render_template('result.html', prediction='Error: Missing or invalid input data. Please check your address and date/time.')

        # Select features
        X = final.iloc[:, [1, 2, 3, 4, 6, 10, 11]].values

        # Predict
        my_prediction = rfc.predict(X)

        # Interpret prediction
        if my_prediction[0][0] == 1:
            my_prediction = 'Predicted crime : Act 379-Robbery'
        elif my_prediction[0][1] == 1:
            my_prediction = 'Predicted crime : Act 13-Gambling'
        elif my_prediction[0][2] == 1:
            my_prediction = 'Predicted crime : Act 279-Accident'
        elif my_prediction[0][3] == 1:
            my_prediction = 'Predicted crime : Act 323-Violence'
        elif my_prediction[0][4] == 1:
            my_prediction = 'Predicted crime : Act 302-Murder'
        elif my_prediction[0][5] == 1:
            my_prediction = 'Predicted crime : Act 363-Kidnapping'
        else:
            my_prediction = 'Place is safe — no crime expected at that timestamp.'

    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
