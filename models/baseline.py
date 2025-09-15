from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_baseline_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax') # 6 activities in HAR dataset
    ])
    return model
