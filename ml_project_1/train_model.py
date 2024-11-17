import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import dump
import os

# Load dataset
data = pd.read_csv(r'B:\ml_project_1\data\house_data.csv')

# Drop rows with missing values
data = data.dropna()

# Define features and target
X = data[['Size', 'Bedrooms']]
y = data['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
os.makedirs(r'B:\ml_project_1\model', exist_ok=True)
dump(model, r'B:\ml_project_1\model\trained_model.pkl')
print("Model trained and saved.")
