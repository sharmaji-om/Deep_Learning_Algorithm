import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import time

# Constants
K_FOLDS = 2
EPOCHS = 50
BATCH_SIZE = 32

# Load Dataset
df = pd.read_csv(r"E:\Student_performance_data.csv")
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

# Convert labels to numerical format
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create K-Fold cross-validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Create directory to save confusion matrices
os.makedirs("confusion_matrices", exist_ok=True)

# Define Neural Network Model
def create_bp_nn(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training and Evaluation
results = []
timing_results = []
fold_idx = 1
num_classes = len(np.unique(y))

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = create_bp_nn(X.shape[1], num_classes)
    
    start_train_time = time.time()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    train_time = time.time() - start_train_time
    
    start_test_time = time.time()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    test_time = time.time() - start_test_time
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save timing results
    timing_results.append({
        'Fold': fold_idx,
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time,
        'Total Time (s)': train_time + test_time
    })
    
    # Save results
    results.append({
        'Fold': fold_idx,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time
    })
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Fold {fold_idx} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrices/fold_{fold_idx}.png')
    plt.close()
    
    fold_idx += 1

# Save Results to CSV
pd.DataFrame(results).to_csv('metrics.csv', index=False)
pd.DataFrame(timing_results).to_csv('time.csv', index=False)
print(results)
print("Training and evaluation completed successfully!")