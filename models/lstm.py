from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')
    ])
    return model
