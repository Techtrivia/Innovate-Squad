from flask import Flask, request, redirect, url_for, render_template, flash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
import csv
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flash messages

# Set up Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirects to login page if not authenticated

# Define the CSV file path for user data
csv_file_path = 'user_data.csv'

# Define the CSV file path for crime reports
report_csv_path = 'crime_reports.csv'

# Ensure the user data CSV file exists with headers
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Username', 'Password', 'Email'])  # CSV headers

# Ensure the crime reports CSV file exists with headers
if not os.path.exists(report_csv_path):
    with open(report_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Phone Number', 'Location', 'Crime Type', 'Description', 'Attachment'])  # Report headers

# Ensure the 'uploads' directory exists for storing attachments
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Load user callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            if row[0] == user_id:
                return User(user_id, row[0])
    return None

# Load the trained Random Forest model and the label encoder
def load_model_and_encoder():
    model = joblib.load('flask_app/models/random_forest_model.pkl')
    label_encoder = joblib.load('flask_app/models/label_encoder.pkl')
    return model, label_encoder

# Registration route
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Append the user data to the CSV file
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([username, password, email])

        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check CSV file for matching credentials
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if row[0] == username and row[1] == password:
                    user = User(username, username)
                    login_user(user)
                    flash('Login successful!')
                    return redirect(url_for('home'))

        flash('Invalid username or password. Please try again.')
        return redirect(url_for('login'))

    return render_template('login.html')

# Home route
@app.route('/home')
@login_required
def home():
    return render_template('home.html')

# About route
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# Report form route
@app.route('/report', methods=['GET'])
@login_required
def report():
    return render_template('report.html')

# Route to handle report form submission
@app.route('/report_crime', methods=['POST'])
@login_required
def report_crime():
    if request.method == 'POST':
        name = request.form['name']
        phone_number = request.form['phone_number']
        location = request.form['location']
        crime_type = request.form['crime_type']
        description = request.form['description']
        
        # Handle file attachment (optional)
        attachment = request.files['attachment']
        attachment_filename = None
        if attachment and attachment.filename:
            # Sanitize the filename and save it to the 'uploads' directory
            attachment_filename = os.path.join(uploads_dir, attachment.filename)
            attachment.save(attachment_filename)

        # Append the report data to the CSV file
        with open(report_csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, phone_number, location, crime_type, description, attachment_filename or "No Attachment"])

        print('Crime report submitted successfully!')
        return redirect(url_for('home'))

# Prediction route for crime prediction
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    model, label_encoder = load_model_and_encoder()
    
    if request.method == 'POST':
        # Get inputs from the user (State, Year, and Crime Type)
        state = request.form['state']
        year = int(request.form['year'])
        crime_type = request.form['crime_type']

        # Encode the inputs using the saved label encoder
        state_encoded = label_encoder.transform([state])[0]
        crime_type_encoded = label_encoder.transform([crime_type])[0]

        # Make prediction using the trained model
        prediction = model.predict([[state_encoded, year, crime_type_encoded]])

        return render_template('predict.html', prediction=prediction[0])

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
