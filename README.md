# Crime Prediction Project

## Description

This project is a web application designed to predict the likelihood of certain types of crimes occurring at a specific location and time within a given area (originally focused on Indore, India). It uses a Random Forest Classifier model trained on historical crime data. The application provides a user interface to input location and time, and then displays the predicted crime type or indicates if the area is predicted to be safe.

The project also includes features for crime analysis visualization (though the backend logic for this might be part of `work.html` and not fully detailed in `app.py`'s prediction logic).

## Features

-   Predicts one of six crime types (Robbery, Gambling, Accident, Violence, Murder, Kidnapping) or "safe".
-   Accepts location (address string) and a specific date/time as input.
-   Uses geocoding to convert addresses to latitude/longitude.
-   Web interface built with Flask and HTML/CSS/JavaScript.
-   Includes a script to retrain the prediction model.

## Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
-   **Geocoding:** Geopy
-   **Frontend:** HTML, CSS, JavaScript (with Bootstrap and various JS libraries for UI components)
-   **Data Storage:** CSV file (`data.csv`) for training data.

## Project Structure

```
/
├── app.py                # Main Flask application
├── train.py              # Script to retrain the ML model
├── data.csv              # Dataset for training the model
├── requirements.txt      # Python dependencies for the project
├── requirements_3.10.txt # Dependencies for Python 3.10 (can be consolidated into requirements.txt)
├── model/
│   └── rf_model          # Saved (trained) RandomForestClassifier model
├── static/               # Static assets (CSS, JS, images, fonts)
│   ├── css/
│   ├── js/
│   ├── images/
│   └── ...
├── templates/            # HTML templates for the web interface
│   ├── index.html        # Main page for crime prediction input
│   ├── result.html       # Page to display prediction results
│   ├── work.html         # Page for crime analysis (visualizations)
│   ├── about.html        # About page
│   └── contact.html      # Contact page
├── RAW DATA/             # Directory for raw data files (potentially for preprocessing)
├── Analysis/             # Directory for analysis outputs (images, etc.)
└── README.md             # This file
```

## Setup and Installation

### Prerequisites

-   Python 3.10.12 (as per the latest update plan)
-   `pip` (Python package installer)
-   A virtual environment manager (e.g., `venv`) is highly recommended.

### Steps

1.  **Clone the Repository**
    Clone the project from GitHub:
    ```bash
    git clone https://github.com/Aryan-amz/Crime-Prediction.git
    cd Crime-Prediction
    ```

2.  **Create and Activate a Virtual Environment**
    It's best practice to use a virtual environment to manage project dependencies.
    ```bash
    python3.10 -m venv venv
    ```
    Activate the environment:
    -   Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    Install the required Python packages using the `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Data

-   The primary dataset used for training the model is `data.csv`. This file should contain historical crime records with timestamps, crime types (as one-hot encoded columns like `act379`, `act13`, etc.), latitude, and longitude.
-   The `RAW DATA/` directory may contain original data files if `data.csv` is a preprocessed version.

## Model Training

The machine learning model (RandomForestClassifier) can be retrained using the provided `train.py` script. This is necessary if:
-   The `data.csv` file is updated.
-   You want to experiment with different model parameters.
-   You encounter compatibility issues with the saved model after library updates (e.g., scikit-learn version changes).

To retrain the model:
1.  Ensure your virtual environment is activated and all dependencies are installed.
2.  Run the training script from the project's root directory:
    ```bash
    python train.py
    ```
    This script will:
    -   Load data from `data.csv`.
    -   Perform feature engineering (extracting date/time components).
    -   Train a new RandomForestClassifier.
    -   Save the trained model to `model/rf_model`, overwriting the existing one.

## Running the Application

1.  **Activate the Virtual Environment** (if not already active):
    -   Windows: `.\venv\Scripts\activate`
    -   macOS/Linux: `source venv/bin/activate`

2.  **Run the Flask Application:**
    Navigate to the project's root directory and run:
    ```bash
    python app.py
    ```
    The application will typically start a development server.

3.  **Access the Web Interface:**
    Open your web browser and go to:
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

    You should see the main page where you can input a location and timestamp to get a crime prediction.

## Notes

-   The geocoding service (Nominatim) requires an internet connection to convert addresses to coordinates.
-   The accuracy of predictions depends heavily on the quality and representativeness of the training data in `data.csv`.
-   The `datetime-local` input in `index.html` expects users to input time in their local timezone. The application currently does not perform explicit timezone conversions, so consistency in time input is important.
