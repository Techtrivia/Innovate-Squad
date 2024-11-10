import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.preprocessing import LabelEncoder

# Define model directory and path
model_dir = 'models'
model_path = os.path.join(model_dir, 'crime_predictor.pkl')

# Create the directory if it does not exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Check if model exists, otherwise train and save
if not os.path.exists(model_path):
    print("Model not found, training a new model...")
    
    # Load the dataset using relative path
    data_path = os.path.join('dataset', 'CrimesOnWomenData.csv')
    data = pd.read_csv(data_path)
    
    # Print the columns for debugging
    print("Columns in the dataset:", data.columns.tolist())
    
    # Prepare the data
    X = data[['Year', 'State']]
    y = data['Rape']
    
    # Encode categorical variables
    le = LabelEncoder()
    X['State'] = le.fit_transform(X['State'])
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, model_path)
    print("Model trained and saved.")
else:
    print("Model already exists.")

