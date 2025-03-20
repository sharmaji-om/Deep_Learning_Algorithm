import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os

# Set global K-Fold value
K_FOLDS = 2

# Load dataset
df = pd.read_csv(r"E:\part-00001_preprocessed_dataset.csv")
df = df.sample(frac=0.2, random_state=42)
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

X = df.drop(columns=['label']).values
y = df['label'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create directories
os.makedirs("confusion_matrices", exist_ok=True)

# Define Bi-GRU model
def create_bi_gru_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(GRU(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
results = []
timing_results = []
num_classes = len(np.unique(y))
input_shape = (X.shape[1], 1)

# Iterate over folds
for fold_idx, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test = X_train.reshape(-1, X.shape[1], 1), X_test.reshape(-1, X.shape[1], 1)

    model = create_bi_gru_model(input_shape, num_classes)
    
    start_train_time = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    train_time = time.time() - start_train_time
    
    start_test_time = time.time()
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_time = time.time() - start_test_time
    
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'Fold': fold_idx,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    timing_results.append({
        'Fold': fold_idx,
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time,
        'Total Time (s)': train_time + test_time
    })
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Bi-GRU - Fold {fold_idx} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrices/bi_gru_fold_{fold_idx}.png")
    plt.close()

# Save results
timing_df = pd.DataFrame(timing_results)
timing_df.to_csv("bi_gru_time.csv", index=False)

results_df = pd.DataFrame(results)
results_df.to_csv("bi_gru_metrics.csv", index=False)
print(results_df)
