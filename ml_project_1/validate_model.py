from joblib import load
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load the trained model
model = load(r'B:\ml_project_1\model\trained_model.pkl')

# Load the data
data = pd.read_csv(r'B:\ml_project_1\data\house_data.csv')

X = data[['Size', 'Bedrooms']]
y = data['Price']

# Make predictions and evaluate
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print(f"Model Mean Squared Error: {mse}")
