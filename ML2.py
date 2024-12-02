import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = './SensorData/30secondbehaviour.csv'
data = pd.read_csv(file_path)

# Preprocessing: Binarize the sensor values
X = data.iloc[:, :-1].applymap(lambda x: 1 if x < 0.8 else 0)  # Binarize LOS sensor values
y = data['y']  # Behavior labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Classification Report
target_names = ['Cycloidal', 'Aggregation', 'Dispersal']
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
print("Classification Report:\n", classification_rep)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = rf.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=X.columns)
plt.xlabel('Sensor Index')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()
