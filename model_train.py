import numpy as np
import pandas as pd 
import joblib
import tensorflow as tf #used for deep learning
from tensorflow import keras # used for data manipulation and analysis
from sklearn.model_selection import train_test_split # used for splitting the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler # used for feature scaling and normalization
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # used for evaluating the model's performance
from sklearn.model_selection import GridSearchCV # used for hyperparameter tuning
from sklearn.pipeline import Pipeline # used for creating a pipeline for 

#import csv file
df = pd.read_csv('heart-disease.csv')
# Display the first few rows of the dataset
print(df.head())

# Step 2: Prepare Data (Select relevant features)yy
X = df.drop(columns=["target"])  # Features for AI training
y = df["target"]  # Labels (1 = high risk, 0 = low risk)

# Step 3: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#**Save the fitted scaler for future use**
joblib.dump(scaler, "model/scaler.pkl")

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Create Deep Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    keras.layers.Dropout(0.3),  # Prevent overfitting
    keras.layers.Dense(32, activation='relu'),  # Second hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification (1 = High Risk, 0 = Low Risk)
])

# Step 6: Compile & Train Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Step 7: Save Model for Deployment
model.save("model/heart_disease_model.h5")  
# Save trained AI model
print("Model Training Complete! Saved as model/heart_disease_model.h5")
