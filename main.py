import argparse
from utils.data_loader import load_data
from models.baseline import build_baseline_model
from models.cnn import build_cnn_model
from models.lstm import build_lstm_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch as tf

def main(model_type):
    print(f"Running HAR with {model_type.upper()} model...")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    input_shape = (X_train.shape[1], 1)
    
    # Select model
    if model_type == "baseline":
        model = build_baseline_model(X_train.shape[1])
    elif model_type == "cnn":
        model = build_cnn_model(input_shape)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    elif model_type == "lstm":
        model = build_lstm_model(input_shape)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    else:
        raise ValueError("Invalid model type. Choose baseline, cnn, or lstm.")
    
    # Compile & train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", help="baseline | cnn | lstm")
    args = parser.parse_args()
    main(args.model)
