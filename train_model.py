import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load your updated cancer dataset
data = pd.read_csv('cancer_data_for_scaling.csv')
data = data.drop(['id'], axis=1)

X = data.drop(['diagnosis'], axis=1)  # Features
y = data['diagnosis']  # Target variable (0 for benign, 1 for malignant)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print a classification report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

# Save the trained model using pickle
with open('cancer_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
