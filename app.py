from flask import Flask, redirect, render_template, request, url_for
import MySQLdb
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import logging
import os
import pickle

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Define the path to the model file
model_file_path = r'F:\Final Project (8th Sem)\Project\LightGBM.pkl'

# Load the model during Flask app startup
def load_model():
    global model
    if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 0:
        try:
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading the model: {e}")
            model = None
    else:
        logging.error("Model file not found or empty.")
        model = None

load_model()

# Configure MySQL connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_1'

# Establish MySQL connection
mysql = MySQLdb.connect(host=app.config['MYSQL_HOST'],
                        user=app.config['MYSQL_USER'],
                        password=app.config['MYSQL_PASSWORD'],
                        db=app.config['MYSQL_DB'])

@app.route('/', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        cursor = mysql.cursor()
        cursor.execute('''INSERT INTO flask_1 (username, email, password) VALUES (%s, %s, %s)''', (username, email, password))
        mysql.commit()
        cursor.close()
        return "Registration successful"
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle form submission and perform login logic
        # Assuming successful login, redirect to index page
        return redirect(url_for('index'))  # Redirect to index page after login
    else:
        # Render the login form template
        return render_template('login.html')


@app.route("/index", methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def prediction():
    # Load the data
    df_train = pd.read_csv(r"C:\data science Internship\pythonProject1\Real Estate Price Prediction\output\cleaned_data_without_outliers.csv")

    # Specify the columns to use for prediction
    feature_columns = ['Bedrooms1', 'Baths', 'Balcony', 'total_sqft', 'Price_per_SQFT']

    # Preprocess the 'Balcony' column
    df_train['Balcony'] = df_train['Balcony'].replace({'Yes': 1, 'No': 0}).astype(int)  # Convert 'Yes'/'No' to numeric (1/0)

    # Drop rows with missing values in the selected feature columns
    df_train.dropna(subset=feature_columns, inplace=True)  # Drop rows with missing values

    # Extract features and target variable
    X = df_train[feature_columns]
    y = df_train['Price']

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # Define LightGBM parameters
    parameters = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'early_stopping_rounds': 50
    }

    # Train the LightGBM model
    model_lgbm = lgb.train(params=parameters,
                           train_set=train_data,
                           valid_sets=[train_data, valid_data],
                           num_boost_round=5000)

    # Retrieve user inputs from the form
    user_input = {
        'Bedrooms1': int(request.form['Bedrooms1']),
        'Baths': int(request.form['Baths']),
        'Balcony': 1 if request.form['Balcony'].lower() == 'yes' else 0,  # Convert 'Yes'/'No' to numeric (1/0)
        'total_sqft': int(request.form['total_sqft']),
        'Price_per_SQFT': int(request.form['Price_per_SQFT'])
    }

    # Convert user inputs to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Make predictions on user input data
    predicted_price = model_lgbm.predict(user_df)

    # Extract the scalar prediction value (assuming one prediction result)
    predicted_price_scalar = predicted_price[0]

    # Render the prediction result in the template
    return render_template('prediction.html', prediction=predicted_price_scalar)

if __name__ == '__main__':
    app.run(debug=True)
