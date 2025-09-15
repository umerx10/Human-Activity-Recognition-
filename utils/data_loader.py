import pandas as pd
import os
from tensorflow.keras.utils import to_categorical

def load_data(dataset_path="UCI HAR Dataset"):
    # File paths
    X_train_path = os.path.join(dataset_path, "train", "X_train.txt")
    y_train_path = os.path.join(dataset_path, "train", "y_train.txt")
    X_test_path  = os.path.join(dataset_path, "test", "X_test.txt")
    y_test_path  = os.path.join(dataset_path, "test", "y_test.txt")

    # Load with pandas (space-separated)
    X_train = pd.read_csv(X_train_path, sep='\s+', header=None)
    y_train = pd.read_csv(y_train_path, sep='\s+', header=None)
    X_test  = pd.read_csv(X_test_path,  sep='\s+', header=None)
    y_test  = pd.read_csv(y_test_path,  sep='\s+', header=None)

    # Convert to numpy
    X_train, X_test = X_train.values, X_test.values
    y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

    # One-hot encode labels (6 classes → [0..5])
    y_train = to_categorical(y_train - 1, num_classes=6)
    y_test  = to_categorical(y_test - 1,  num_classes=6)

    return X_train, X_test, y_train, y_test
