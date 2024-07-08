import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to preprocess data
def preprocess_data(df):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Warehouse_block'] = label_encoder.fit_transform(df['Warehouse_block'])
    df['Mode_of_Shipment'] = label_encoder.fit_transform(df['Mode_of_Shipment'])
    df['Product_importance'] = label_encoder.fit_transform(df['Product_importance'])
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    # Define feature variables and target variable
    X = df.drop(['ID', 'Reached.on.Time_Y.N'], axis=1)  # Features
    y = df['Reached.on.Time_Y.N']  # Target
    
    return X, y

# Load the Dataset
file_path = 'Ecomm.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Preprocess the Data
X, y = preprocess_data(data)

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Set Scaling Model
scl = RobustScaler()
X_train_scaled = scl.fit_transform(X_train)
X_test_scaled = scl.transform(X_test)

# Optimize Prior using StratifiedKFold
priors_range = np.arange(0.1, 1, 0.001)
scores = []

# StratifiedKFold for better class distribution in each fold
skf = StratifiedKFold(n_splits=7)

for prior in priors_range:
    clf = GaussianNB(priors=[prior, 1 - prior])
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, n_jobs=-1, scoring='accuracy')
    scores.append(np.mean(cv_scores))

optimal_prior = priors_range[np.argmax(scores)]
print(f'Optimal Prior: {optimal_prior}')

# Train Naive Bayes Classifier with optimal prior
clf = GaussianNB(priors=[optimal_prior, 1 - optimal_prior])
clf.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = clf.predict(X_test_scaled)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Get unique classes from y_test and y_pred to set axis labels
labels = np.unique(y_test)
sns.set(font_scale=1.2)  # Adjust font scale if needed

# Visualize Results
# Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print Accuracy and Classification Report
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
