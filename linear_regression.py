import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Generate Data
np.random.seed(42)
hours = np.random.uniform(1, 10, 50)  # Random study hours between 1 and 10
scores = 5 * hours + np.random.normal(0, 5, 50)  # Exam score formula with noise
data = pd.DataFrame({'Hours': hours, 'Scores': scores})

# Step 3: Visualize the Data
plt.scatter(data['Hours'], data['Scores'], color='blue', label='Data Points')
plt.title('Study Hours vs Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()

# Step 4: Prepare the Data
X = data[['Hours']]
y = data['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model parameters
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 7: Visualize the Regression Line
plt.scatter(data['Hours'], data['Scores'], color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()

# Step 8: Make Predictions
new_hours = np.array([[8], [9.5]])  # New data points

new_hours_df = pd.DataFrame(new_hours, columns=['Hours'])
predicted_scores = model.predict(new_hours_df)
print(f"Predicted scores for {new_hours_df['Hours'].values} hours: {predicted_scores}")
