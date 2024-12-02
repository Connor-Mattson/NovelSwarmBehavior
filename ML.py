# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import os

# # Define the column names
# column_names = ["Time Stamp", "BotName", "LOS_sensor", "bot_speed_x", "bot_speed_y", "Pattern"]

# # Load data from each CSV file and assign column names
# aggregation_path = './SensorData/Aggregation.csv'
# cycloidal_path = './SensorData/Cycloidal_Data.csv'
# dispersal_path = './SensorData/Dispersal.csv'

# aggregation_df = pd.read_csv(aggregation_path, names=column_names, skiprows=2)
# cycloidal_df = pd.read_csv(cycloidal_path, names=column_names, skiprows=2)
# dispersal_df = pd.read_csv(dispersal_path, names=column_names, skiprows=2)

# # Combine data and assign behavior labels
# sensor_data_df = pd.concat([aggregation_df, cycloidal_df, dispersal_df], ignore_index=True)

# # Group data from bots 0-29 as one time instance
# sensor_data_df['time_instance'] = sensor_data_df.index // 30

# # Aggregate features to create swarm-level features
# swarm_data_df = sensor_data_df.groupby(['time_instance', 'Pattern']).agg({
#     'LOS_sensor': ['mean', 'sum'],
#     'bot_speed_x': ['mean', 'std'],
#     'bot_speed_y': ['mean', 'std']
# }).reset_index()

# # Flatten the multi-level columns
# swarm_data_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in swarm_data_df.columns]

# # Prepare features and target variable
# X = swarm_data_df.drop(columns=['time_instance_', 'Pattern_'])
# y = swarm_data_df['Pattern_']

# # Fill NaN values with column means
# X = X.fillna(X.mean())

# # Splitting data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# # Initialize and train the RandomForest Classifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Predict class probabilities for each sample
# proba = rf.predict_proba(X_test)

# # Predict the class labels
# y_pred = rf.predict(X_test)

# # Classification report
# target_names = ['Cycloidal', 'Aggregation', 'Dispersal']
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Feature Importance Visualization
# feature_importances = rf.feature_importances_
# plt.figure(figsize=(10, 6))
# plt.bar(range(X.shape[1]), feature_importances, tick_label=X.columns)
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.title('Feature Importance in Random Forest Classifier')
# plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Define the column names
column_names = ["Time Stamp", "BotName", "LOS_sensor", "bot_speed_x", "bot_speed_y", "Pattern"]

# Load data from each CSV file and assign column names
aggregation_path = './SensorData/Aggregation.csv'
cycloidal_path = './SensorData/Cycloidal_Data.csv'
dispersal_path = './SensorData/Dispersal.csv'

aggregation_df = pd.read_csv(aggregation_path, names=column_names, skiprows=2)
cycloidal_df = pd.read_csv(cycloidal_path, names=column_names, skiprows=2)
dispersal_df = pd.read_csv(dispersal_path, names=column_names, skiprows=2)

# Combine data and assign behavior labels
sensor_data_df = pd.concat([aggregation_df, cycloidal_df, dispersal_df], ignore_index=True)

# Group data from bots 0-29 as one time instance
sensor_data_df['time_instance'] = sensor_data_df.index // 30

# Aggregate features to create swarm-level features using only LOS_sensor
swarm_data_df = sensor_data_df.groupby(['time_instance', 'Pattern']).agg({
    'LOS_sensor': ['mean', 'sum']
}).reset_index()

# Flatten the multi-level columns
swarm_data_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in swarm_data_df.columns]

# Prepare features and target variable
X = swarm_data_df.drop(columns=['time_instance_', 'Pattern_'])
y = swarm_data_df['Pattern_']

# Fill NaN values with column means
X = X.fillna(X.mean())

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initialize and train the RandomForest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict class probabilities for each sample
proba = rf.predict_proba(X_test)

# Predict the class labels
y_pred = rf.predict(X_test)

# Classification report
target_names = ['Cycloidal', 'Aggregation', 'Dispersal']
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Visualization
feature_importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances, tick_label=X.columns)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()
