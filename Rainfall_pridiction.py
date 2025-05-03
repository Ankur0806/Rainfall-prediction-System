import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate a synthetic dataset (replace this with your real dataset)
def create_synthetic_data(n_samples=1000):
    data = {
        'Temperature': np.random.uniform(10, 40, n_samples),  # Celsius
        'Humidity': np.random.uniform(20, 100, n_samples),    # Percentage
        'Wind_Speed': np.random.uniform(0, 20, n_samples),    # km/h
        'Pressure': np.random.uniform(980, 1030, n_samples),  # hPa
    }
    df = pd.DataFrame(data)
    
    # Simple rule for rainfall: high humidity and lower pressure increase rain likelihood
    df['Rain'] = np.where((df['Humidity'] > 70) & (df['Pressure'] < 1000), 'Rain', 'No Rain')
    return df

# Load or create dataset
df = create_synthetic_data()
# If you have a real dataset, uncomment and modify the line below:
# df = pd.read_csv('your_weather_data.csv')

# Step 2: Data Preprocessing
print("Dataset Preview:")
print(df.head())

# Convert categorical target to binary (0 = No Rain, 1 = Rain)
df['Rain'] = df['Rain'].map({'No Rain': 0, 'Rain': 1})

# Features and target
X = df[['Temperature', 'Humidity', 'Wind_Speed', 'Pressure']]
y = df['Rain']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = rf_model.predict(X_test)

# Step 5: Evaluate the Model
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# F1 Score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 6: Feature Importance (Optional)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Step 7: Example Prediction
example = np.array([[25, 80, 5, 990]])  # Temp=25°C, Humidity=80%, Wind=5km/h, Pressure=990hPa
prediction = rf_model.predict(example)
print(f"\nExample Prediction (0 = No Rain, 1 = Rain): {prediction[0]}")