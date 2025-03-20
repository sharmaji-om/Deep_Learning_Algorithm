import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os

# Set parameters
K_FOLDS = 2
EPOCHS = 10
BATCH_SIZE = 32

# Read dataset
df = pd.read_csv(r"E:\part-00001_preprocessed_dataset.csv")
df = df.sample(frac=0.2, random_state=42)
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

# Prepare data
X = df.drop(columns=['label']).values
y = tf.keras.utils.to_categorical(df['label'], num_classes)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
results = []

def build_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False, activation='tanh')),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform K-Fold Cross-Validation
for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = build_model((X_train.shape[1], 1), num_classes)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    report = classification_report(y_test_labels, y_pred, output_dict=True)
    results.append(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Fold {fold_idx+1} Confusion Matrix')
    plt.savefig(f'confusion_matrices/BiLSTM_fold_{fold_idx+1}.png')
    plt.close()

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('BiLSTM_results.csv', index=False)

print('Bi-LSTM MSCBL-ADN training completed.')
