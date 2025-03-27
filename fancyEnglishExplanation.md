
---

# Rain Prediction AI Model & Web App Report

## Abstract
This report outlines the development of a rain prediction system using historical weather data from Meteostat. The project encompasses the entire machine learning pipeline—from data collection and preprocessing, model training, and evaluation, to deploying a web application using Flask. A Random Forest Regressor was chosen as the primary algorithm, and various techniques such as hyperparameter tuning and post-processing (clipping negative predictions) were employed to optimize performance. The final system provides rainfall predictions for the upcoming days via a user-friendly web interface.

---

## 1. Introduction
Predicting rainfall is an essential task for weather forecasting and agricultural planning. With the increasing availability of historical weather data and advancements in machine learning, it is now feasible to build predictive models that can forecast rainfall based on multiple meteorological features. This project aimed to create an AI model that predicts rainfall and deploys it as a web app, making the results accessible to users in real time.

---

## 2. Environment Setup

### 2.1 Tools and Technologies
- **Python:** The primary programming language used for data processing, model training, and web development.
- **Jupyter Notebook:** An interactive environment used for coding, exploratory data analysis, and model development.
- **Flask:** A lightweight Python web framework used to create and deploy the web application.
- **Libraries:** Pandas, NumPy, scikit-learn, matplotlib, and pickle for model serialization.

### 2.2 Installation
The required libraries were installed using pip:
```bash
pip install jupyter pandas numpy scikit-learn matplotlib flask
```
Jupyter Notebook was used to interactively develop and test the data pipeline and model.

---

## 3. Data Collection and Preparation

### 3.1 Data Acquisition
Historical weather data for Minna, Niger State was sourced from [Meteostat.net](https://meteostat.net/en/). The dataset contained daily records for 10 years, including variables such as:
- Date
- Average temperature (`tavg`)
- Minimum temperature (`tmin`)
- Maximum temperature (`tmax`)
- Precipitation (`prcp`)
- Wind direction (`wdir`)
- Wind speed (`wspd`)
- Pressure (`pres`)
- Additional columns (e.g., snow, tsun) that were either irrelevant or entirely missing.

### 3.2 Data Cleaning and Preprocessing
- **Column Removal:** Columns with all missing values (e.g., `snow`, `wpgt`, `tsun`) were removed.
- **Missing Value Treatment:** Missing values in the remaining columns were handled using forward-fill and backward-fill techniques:
  ```python
  df = df.ffill().bfill()
  ```
- **Date Conversion:** The date column was converted from string to datetime format to facilitate time-series analysis.
- **Exploratory Data Analysis:** Visualizations (e.g., plotting temperature trends) were performed to ensure data consistency and to detect any anomalies.

---

## 4. Model Building

### 4.1 Feature Selection
The following features were chosen as predictors for rainfall:
- `tavg`, `tmin`, `tmax` (temperature metrics)
- `wdir`, `wspd` (wind characteristics)
- `pres` (atmospheric pressure)
The target variable was `prcp` (precipitation).

### 4.2 Initial Model Training
An initial attempt was made using a Linear Regression model, but it resulted in very low accuracy. This indicated that a more robust model was required.

### 4.3 Model Improvement: Random Forest Regressor
- **Model Choice:** A Random Forest Regressor was selected for its ability to capture complex, nonlinear relationships.
- **Training:** The model was trained using scikit-learn after splitting the data into training (80%) and testing (20%) sets.
- **Hyperparameter Tuning:** The model was further optimized using RandomizedSearchCV to fine-tune parameters like the number of estimators and tree depth.
- **Post-Processing:** Since the regression output could produce negative values (which are not realistic for rainfall), a post-processing step was added to clip all negative predictions to 0:
  ```python
  predictions = np.clip(predictions, 0, None)
  ```

### 4.4 Model Persistence
Once the model was trained and optimized, it was serialized (saved) using Python's pickle module:
```python
with open("rain_model.pkl", "wb") as file:
    pickle.dump(model, file)
```
This saved model is later loaded by the web application.

---

## 5. Web Application Development

### 5.1 Flask Setup
A Flask web application was built to serve the predictions. The application performs the following tasks:
- Loads the saved model.
- Generates dummy future weather data (for demonstration) and computes predictions for upcoming days.
- Provides two main routes:
  - **Home (`/`):** Displays the predicted rainfall for the upcoming days (initially limited to 7 days).
  - **Predict (`/predict`):** Accepts a user-selected date from a form and returns the corresponding rainfall prediction as JSON.

### 5.2 Code Implementation
The `app.py` file contains the following key components:
- **Model Loading:** Using pickle to load `rain_model.pkl`.
- **Date Generation:** Creating a list of future dates in `YYYY-MM-DD` format.
- **Dummy Data Generation:** Creating a DataFrame of random values (with correct feature names) to simulate future weather data.
- **Prediction Generation:** Making predictions using the Random Forest model and ensuring no negative values are returned.
- **Routes:**  
  - The home route uses `render_template` to display predictions.
  - The predict route processes user input and returns predictions in JSON format.

### 5.3 User Interface
A simple HTML template (`index.html`) was created in the `templates` folder. This template:
- Displays the upcoming predictions.
- Provides a form for the user to select a date and fetch the predicted rainfall for that specific date.
- Uses JavaScript to asynchronously fetch predictions without reloading the page.

---

## 6. Discussion and Challenges

### 6.1 Key Achievements
- **End-to-End Pipeline:** The project successfully demonstrates an entire pipeline from data preprocessing, model training, and evaluation to deployment as a web application.
- **Model Selection:** Transitioning from a basic Linear Regression to a more powerful Random Forest Regressor improved the model’s performance.
- **Practical Deployment:** Integrating the model into a Flask web app provides a real-world interface for users to interact with the predictions.

### 6.2 Challenges Encountered
- **Data Quality:** Handling missing values and ensuring that the features were correctly formatted were significant early challenges.
- **Model Output:** The model initially produced negative rainfall predictions, which required a post-processing step to correct.
- **Feature Consistency:** Maintaining consistency between training feature names and prediction inputs was essential to avoid errors.
- **User Interface:** Limiting the predictions displayed on the homepage (first 7 days) while generating a broader set of predictions (365 days) required careful slicing and presentation logic.

---

## 7. Conclusion and Future Work
The project successfully developed a rain prediction model using a Random Forest Regressor, integrated it into a Flask web application, and demonstrated a complete machine learning workflow. Although the current model uses dummy data for future weather predictions, future work will focus on:
- Integrating real-time weather forecast data.
- Enhancing the user interface with improved design elements.
- Exploring additional models and feature engineering techniques to further improve prediction accuracy.

This project serves as a robust demonstration of how machine learning can be applied to real-world problems and how such models can be deployed in a web environment for practical use.

---

## 8. References
- Meteostat (https://meteostat.net/en/)
- Scikit-learn documentation
- Flask documentation
