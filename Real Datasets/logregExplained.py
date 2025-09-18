import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ...existing code...
from sklearn.model_selection import train_test_split  # only for splitting
from sklearn.preprocessing import StandardScaler      # only for scaling
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

print("Working directory:", os.getcwd())

"""
os: For file system operations (checking current directory)
numpy: Mathematical operations and array handling
pandas: Reading CSV files and data manipulation
matplotlib: Creating plots and visualizations
sklearn: Pre-built tools for data splitting, scaling, and evaluation metrics
The code prints current directory to verify file paths
"""

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

"""
Purpose: Converts any real number into a probability between 0 and 1
Formula: σ(z) = 1/(1 + e^(-z))
Why Important:
    When z is large positive → output ≈ 1 (high probability)
    When z is large negative → output ≈ 0 (low probability)
    When z = 0 → output = 0.5 (neutral)
Use Case: Transforms linear predictions into probabilities for classification
"""

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr              # Learning rate
        self.epochs = epochs      # Number of training iterations
        self.weights = None       # Feature weights (to be learned)
        self.bias = None          # Bias term (to be learned)
        self.losses = []          # Track training progress
        
    '''
    lr (learning rate): How big steps to take when learning (0.01 = small careful steps)
    epochs: How many times to go through the entire dataset (1000 iterations)
    weights: The importance of each feature (starts empty, learned during training)
    bias: A starting point adjustment (like y-intercept in linear regression)
    losses: Track how wrong our predictions are over time
    '''

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            # gradients
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            
            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # compute and store loss (binary cross-entropy)
            loss = - (1/self.m) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))
            self.losses.append(loss)

    '''
    Setup: Get dataset dimensions (m=samples, n=features), start weights at zero
    For each epoch (training iteration):
        Forward Pass: Calculate predictions

            linear_model = X·weights + bias (dot product of features and weights)
            y_pred = sigmoid(linear_model) (convert to probabilities)

        Calculate Gradients: How much to adjust weights

        dw: How much each weight should change
        db: How much bias should change
        
        Update Parameters: Adjust weights and bias

            Move weights in direction that reduces error
            Learning rate controls step size

        Calculate Loss: How wrong our predictions are

            Binary cross-entropy: penalizes confident wrong predictions heavily
    '''

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in proba])

''' 
predict_proba: Returns probability scores (0.0 to 1.0)
predict: Converts probabilities to binary decisions (0 or 1)
    If probability ≥ 50% → predict heart disease (1)
    If probability < 50% → predict no heart disease (0)
'''

df = pd.read_csv("Real Datasets/heart_disease.csv")
df = df.dropna()

'''
Read the heart disease dataset from CSV file
Remove any rows with missing values
'''

if df['cp'].dtype == object:
    df = pd.get_dummies(df, columns=['cp'], drop_first=True)

'''
Check if 'cp' (chest pain type) is text/categorical
Convert categories to numbers using one-hot encoding
Example: if cp has ["typical", "atypical", "none"] → creates binary columns
'''

X = df.drop(columns=['target'])
y = df['target'].values

'''
X: All input features (age, cholesterol, blood pressure, etc.)
y: What we want to predict (0 = no heart disease, 1 = heart disease)
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Split data: 80% for training, 20% for testing
Training data: Teach the model
Test data: Evaluate how well it learned (model never sees this during training)
'''

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
Standardize all features to have mean=0, std=1
Why needed: Age (20-80) and cholesterol (100-400) have different scales
Without scaling: large numbers dominate, small numbers ignored
'''

model_cls = LogisticRegressionScratch(lr=0.01, epochs=5000)
model_cls.fit(X_train_scaled, y_train)
y_pred = model_cls.predict(X_test_scaled)
probas = model_cls.predict_proba(X_test_scaled)

accuracy = np.mean(y_pred == y_test)
print("Classification Accuracy:", accuracy)

'''
Create model with learning rate 0.01, train for 5000 epochs
Train the model on training data
Make predictions on test data
Calculate accuracy: percentage of correct predictions
'''

plt.figure(figsize=(8,5))
plt.plot(model_cls.losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss (Binary Cross-Entropy)")
plt.title("Logistic Regression Training Convergence")
plt.legend()
plt.show()

'''
How the model's error decreases over time
Should trend downward (model getting better)
Flat line at end = model has converged
7b: Confusion Matrix
'''

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Heart Disease", "Heart Disease"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression Scratch")
plt.show()

'''
2x2 grid showing correct vs incorrect predictions
Diagonal = correct predictions
Off-diagonal = mistakes
'''

fpr, tpr, thresholds = roc_curve(y_test, probas)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression Scratch")
plt.legend()
plt.show()

'''
Trade-off between catching heart disease (sensitivity) vs false alarms
AUC score: 0.5 = random guessing, 1.0 = perfect
Curve closer to top-left corner = better model
'''