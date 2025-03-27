# Rain Prediction AI Model & Web App Report

## Abstract
This project demonstrates the end-to-end process of building an AI model to predict rainfall using historical weather data and deploying it as a web application. We start with data collection and preprocessing, move through model development using machine learning techniques, and finish by integrating the model into a Flask web app for interactive predictions. The final model is a Random Forest Regressor that learns from weather features and predicts daily precipitation amounts, with a post-processing step to ensure realistic (non-negative) outputs.

---

## 1. Introduction
Predicting rainfall accurately is crucial for various applications such as agriculture, disaster management, and urban planning. This project leverages historical weather data and machine learning to forecast rainfall. The primary goals are:
- To develop a robust regression model that predicts rainfall based on key meteorological variables.
- To deploy the model in a user-friendly web application, enabling users to view forecasts for upcoming days.

---

## 2. Environment Setup

### 2.1 Tools and Technologies
- **Python:** The core language for data manipulation, modeling, and web development.
- **Jupyter Notebook:** An interactive platform for coding, data analysis, and model experimentation.
- **Flask:** A lightweight Python web framework used for building the web application.
- **Key Libraries:**  
  - **Pandas & NumPy:** For data handling and numerical operations.  
  - **scikit-learn:** For model training and evaluation.  
  - **Matplotlib:** For data visualization.  
  - **pickle:** For saving and loading the trained model.

### 2.2 Installation
The required packages were installed using pip:
```bash
pip install jupyter pandas numpy scikit-learn matplotlib flask
```
Jupyter Notebook was then launched to create and test our scripts interactively.

---

## 3. Data Collection and Preparation

### 3.1 Data Acquisition
Historical weather data for Minna, Niger State was obtained from [Meteostat.net](https://meteostat.net/en/). The dataset spans 10 years and includes key variables such as:
- **Temperature:** Average (`tavg`), minimum (`tmin`), and maximum (`tmax`)
- **Wind:** Direction (`wdir`) and speed (`wspd`)
- **Pressure:** (`pres`)
- **Precipitation:** (`prcp`) – the target variable
Additional columns (e.g., snow, tsun) were present but later removed due to insufficient data.

### 3.2 Data Cleaning and Preprocessing
- **Column Removal:** Irrelevant or entirely empty columns were dropped.
- **Handling Missing Values:** Missing entries were addressed using forward-fill (`ffill()`) and backward-fill (`bfill()`) techniques, ensuring a complete dataset.
- **Data Type Conversion:** The date column was converted from a string to a datetime object, which is essential for time series analysis.
- **Exploratory Analysis:** Visualizations such as line graphs of temperature trends helped confirm that the data was clean and consistent.

---

## 4. Model Development

### 4.1 Feature Selection
From the cleaned dataset, we selected the following predictors:
- **Features:** `tavg`, `tmin`, `tmax`, `wdir`, `wspd`, `pres`
- **Target Variable:** `prcp` (precipitation)

### 4.2 Initial Model Training
An initial attempt using a Linear Regression model resulted in low accuracy, indicating that the relationships between weather features and rainfall were too complex for a simple model.

### 4.3 Advanced Modeling: Random Forest Regressor
- **Model Choice:** A Random Forest Regressor was chosen due to its ability to model nonlinear relationships and handle feature interactions.
- **Training:** The model was trained on 80% of the data (with 20% reserved for testing) using scikit-learn.
- **Hyperparameter Tuning:** Techniques like RandomizedSearchCV were employed to fine-tune the model’s parameters (e.g., number of trees, maximum depth).
- **Post-Processing:** Since regression outputs can be negative—which is not realistic for rainfall—the predictions were clipped using NumPy’s `np.clip` to enforce non-negativity.

### 4.4 Model Saving
Once trained and optimized, the model was serialized (saved) using the pickle module:
```python
with open("rain_model.pkl", "wb") as file:
    pickle.dump(model, file)
```
This saved model can be loaded later for making predictions without retraining.

---

## 5. Web Application Development

### 5.1 Flask Web App Overview
A Flask web application was built to provide an interactive interface for accessing rainfall predictions. The app performs the following functions:
- **Model Loading:** Loads the previously saved model (`rain_model.pkl`).
- **Prediction Generation:** Generates future weather data (dummy data for demonstration) and computes predictions using the model.
- **User Interface:** Displays upcoming predictions (e.g., for the next 7 days) and allows users to input a specific date to retrieve a prediction.

### 5.2 Code Structure
The `app.py` file includes:
- **Route `/`:** Renders an HTML template (`index.html`) that lists the upcoming 7-day predictions.
- **Route `/predict`:** Handles form submissions. Users select a date, and the app returns the corresponding predicted rainfall in JSON format.

### 5.3 User Interface (HTML)
A simple HTML file (`index.html`) was placed in the `templates` folder. This file includes:
- A section displaying the 7-day predictions.
- A form that lets users choose a date, triggering an AJAX request to the `/predict` route.
- JavaScript code to update the interface dynamically without reloading the page.

### 5.4 Deployment
The Flask app was run locally using:
```bash
python app.py
```
Users could then access the application via `http://127.0.0.1:5000`. Future plans include deploying the app to a cloud platform for broader accessibility.

---

## 6. Discussion of Modifications (“Flippings”)

### Data Cleaning and Preprocessing Adjustments
- **Column Removal and Missing Value Handling:**  
  Dropping irrelevant columns and filling missing values ensured the data was complete and consistent, which is crucial for training a reliable model.

### Model Selection and Improvement
- **Switch from Linear Regression to Random Forest:**  
  The initial low performance of Linear Regression highlighted the need for a model capable of capturing more complex patterns. Random Forest was chosen for its robustness and ability to average multiple decision trees.
- **Hyperparameter Tuning:**  
  RandomizedSearchCV was used to optimize parameters, improving the model’s generalization ability on unseen data.

### Post-Processing of Predictions
- **Clipping Negative Values:**  
  Since rainfall cannot be negative, a post-processing step using `np.clip` was introduced to ensure all predictions are realistic.

### Web App Adjustments
- **Limited Display of Predictions:**  
  Although the model generated predictions for a larger range (e.g., 365 days), the home page was designed to display only the first 7 days. This improves usability and focuses the user’s attention on near-term forecasts.
- **Feature Name Consistency:**  
  Throughout the process, special attention was given to ensure that the feature names used during model training were consistent with those used for generating predictions in the web app, preventing errors.

---

## 7. Conclusion and Future Work
This project demonstrates a complete machine learning workflow from data collection and model training to web deployment. The Random Forest Regressor provided a robust approach for rainfall prediction, and the Flask web app offered an accessible interface for end-users. Future enhancements could include:
- Integrating real weather forecast data to replace dummy inputs.
- Experimenting with other models (e.g., neural networks or time-series specific models like LSTMs).
- Improving the UI with modern design frameworks.
- Deploying the app to a cloud service for real-world usage.

---

## 8. References
- Meteostat: [https://meteostat.net/en/](https://meteostat.net/en/)
- Scikit-learn Documentation
- Flask Documentation


