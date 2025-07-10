import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

class LSTMPredictor:
    """
    A class to implement an LSTM model for predicting CPU and memory needs
    based on application characteristics from historical deployment data.

    Inputs (for prediction): OS (Linux/Windows), Architecture (32/64-bit),
                             Base Image Size (GB).
    Outputs: Predicted CPU cores, RAM in GB.
    """

    def __init__(self, look_back=1, lstm_units=64, epochs=50, batch_size=32):
        """
        Initializes the LSTMPredictor with specified parameters.

        Args:
            look_back (int): Number of previous time steps (or features) to use as input.
                             For this problem, it's more about feature-based prediction
                             than time-series, so look_back will be 1, and features
                             will be derived from input characteristics.
            lstm_units (int): Number of LSTM units in the hidden layer.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size for training.
        """
        self.look_back = look_back
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scalers = {} # Scalers for numerical features and outputs
        self.model_cpu = None
        self.model_ram = None
        # Define the ordered list of features expected by the model after one-hot encoding
        self.features_list = [
            'base_image_size_gb',
            'os_Linux', 'os_Windows', 'architecture_32-bit', 'architecture_64-bit'
        ]
        self.output_targets = ['cpu_cores_needed', 'ram_gb_needed']

    def _preprocess_data(self, df: pd.DataFrame):
        """
        Preprocesses the input DataFrame: one-hot encodes categorical features,
        scales numerical features and target variables.
        Ensures consistent feature order for the model.
        """
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['os', 'architecture'], drop_first=False)

        # Ensure all expected feature columns are present, fill missing with 0
        # This is crucial for consistent input shape during training and prediction
        for feature in self.features_list:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0

        # Reorder columns to match self.features_list
        X = df_encoded[self.features_list].values.astype('float32')
        y_cpu = df_encoded['cpu_cores_needed'].values.astype('float32').reshape(-1, 1)
        y_ram = df_encoded['ram_gb_needed'].values.astype('float32').reshape(-1, 1)

        # Scale numerical features and targets
        self.scalers['features'] = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scalers['features'].fit_transform(X)

        self.scalers['cpu_cores_needed'] = MinMaxScaler(feature_range=(0, 1))
        y_cpu_scaled = self.scalers['cpu_cores_needed'].fit_transform(y_cpu)

        self.scalers['ram_gb_needed'] = MinMaxScaler(feature_range=(0, 1))
        y_ram_scaled = self.scalers['ram_gb_needed'].fit_transform(y_ram)

        # Reshape for LSTM: [samples, look_back, features]
        # Since look_back is 1 for this feature-based prediction, it's [samples, 1, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], self.look_back, X_scaled.shape[1])

        return X_reshaped, y_cpu_scaled, y_ram_scaled, X_scaled.shape[1] # Return actual number of features

    def train(self, data_path="data/historical_deployments.csv"):
        """
        Trains the LSTM models for CPU and RAM prediction.
        A dummy data file will be created if the specified path does not exist.

        Args:
            data_path (str): Path to the historical deployment data CSV.
        """
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found at {data_path}. Creating a dummy historical_deployments.csv for training.")
            # Create a dummy file if it doesn't exist
            dummy_data = """os,architecture,base_image_size_gb,cpu_cores_needed,ram_gb_needed
Linux,64-bit,1.2,1,2
Linux,64-bit,1.5,2,4
Windows,64-bit,3.0,2,6
Linux,32-bit,0.8,1,1
Linux,64-bit,2.0,3,8
Windows,64-bit,2.8,2,5
Linux,64-bit,1.0,1,2
Linux,64-bit,1.8,3,7
Windows,32-bit,2.5,1,4
Linux,64-bit,1.3,2,3
Linux,64-bit,1.7,3,6
Windows,64-bit,3.2,4,10
Linux,64-bit,1.1,1,2
Linux,64-bit,2.1,3,7
Windows,64-bit,2.9,2,5
Linux,64-bit,1.4,1,3
Linux,64-bit,1.9,3,8
Windows,32-bit,2.6,2,5
Linux,64-bit,1.6,2,4
Linux,64-bit,2.2,4,9
"""
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, "w") as f:
                f.write(dummy_data)
            print(f"Dummy data saved to {data_path}")

        df = pd.read_csv(data_path)
        X_reshaped, y_cpu_scaled, y_ram_scaled, n_features = self._preprocess_data(df)

        # Split data for training and validation
        X_train, X_val, y_cpu_train, y_cpu_val, y_ram_train, y_ram_val = \
            train_test_split(X_reshaped, y_cpu_scaled, y_ram_scaled, test_size=0.2, random_state=42)

        # Build and train CPU model
        self.model_cpu = Sequential()
        self.model_cpu.add(LSTM(self.lstm_units, activation='relu', input_shape=(self.look_back, n_features)))
        self.model_cpu.add(Dropout(0.2))
        self.model_cpu.add(Dense(1))
        self.model_cpu.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        print("\nTraining LSTM model for CPU prediction...")
        self.model_cpu.fit(X_train, y_cpu_train, epochs=self.epochs, batch_size=self.batch_size,
                           validation_data=(X_val, y_cpu_val), callbacks=[early_stopping], verbose=0)
        print("CPU model training complete.")

        # Build and train RAM model
        self.model_ram = Sequential()
        self.model_ram.add(LSTM(self.lstm_units, activation='relu', input_shape=(self.look_back, n_features)))
        self.model_ram.add(Dropout(0.2))
        self.model_ram.add(Dense(1))
        self.model_ram.compile(optimizer='adam', loss='mse')

        print("Training LSTM model for RAM prediction...")
        self.model_ram.fit(X_train, y_ram_train, epochs=self.epochs, batch_size=self.batch_size,
                          validation_data=(X_val, y_ram_val), callbacks=[early_stopping], verbose=0)
        print("RAM model training complete.")

    def predict(self, os_type: str, architecture: str, base_image_size_gb: float):
        """
        Makes a prediction for CPU cores and RAM in GB based on application characteristics.

        Args:
            os_type (str): Operating system ('Linux' or 'Windows').
            architecture (str): Architecture ('32-bit' or '64-bit').
            base_image_size_gb (float): Size of the base image in GB.

        Returns:
            tuple: (predicted_cpu_cores, predicted_ram_gb)
        """
        if self.model_cpu is None or self.model_ram is None:
            print("Error: Models not trained. Please call train() first.")
            return None, None

        # Create input dictionary and then convert to DataFrame for one-hot encoding
        input_data = {
            'os': [os_type],
            'architecture': [architecture],
            'base_image_size_gb': [base_image_size_gb]
        }
        input_df = pd.DataFrame(input_data)

        # One-hot encode categorical features for input
        input_df_encoded = pd.get_dummies(input_df, columns=['os', 'architecture'], drop_first=False)

        # Prepare a zero-filled array to ensure consistent feature order and presence
        processed_input = np.zeros((1, len(self.features_list)), dtype=np.float32)

        # Populate the processed_input array based on the expected features_list order
        for i, feature in enumerate(self.features_list):
            if feature in input_df_encoded.columns:
                processed_input[0, i] = input_df_encoded[feature].iloc[0]
            # If feature is not in input_df_encoded, it remains 0 (which is correct for one-hot)

        # Scale the input data using the scaler fitted during training
        scaled_input = self.scalers['features'].transform(processed_input)

        # Reshape for LSTM: [samples, look_back, features]
        scaled_input_reshaped = scaled_input.reshape(1, self.look_back, scaled_input.shape[1])

        # Make predictions
        predicted_cpu_scaled = self.model_cpu.predict(scaled_input_reshaped)
        predicted_ram_scaled = self.model_ram.predict(scaled_input_reshaped) # Predict RAM independently

        # Inverse transform predictions to original scale
        # Explicitly cast to Python native types (int and float)
        predicted_cpu_cores = int(max(1, round(self.scalers['cpu_cores_needed'].inverse_transform(predicted_cpu_scaled)[0][0])))
        predicted_ram_gb = float(max(1, round(self.scalers['ram_gb_needed'].inverse_transform(predicted_ram_scaled)[0][0])))

        return predicted_cpu_cores, predicted_ram_gb

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure data directory exists
    # When running this script directly from 'AutoPaaS-X/ai', 'data' is at '../data'
    os.makedirs("../data", exist_ok=True)

    predictor = LSTMPredictor(epochs=50, batch_size=8)
    predictor.train(data_path="../data/historical_deployments.csv")

    print("\n--- Making Predictions ---")
    # Example 1: Linux, 64-bit, small app
    cpu, ram = predictor.predict("Linux", "64-bit", 1.5)
    print(f"Predicted for Linux, 64-bit, 1.5GB: CPU={cpu} cores, RAM={ram} GB")

    # Example 2: Windows, 64-bit, larger app
    cpu, ram = predictor.predict("Windows", "64-bit", 3.0)
    print(f"Predicted for Windows, 64-bit, 3.0GB: CPU={cpu} cores, RAM={ram} GB")

    # Example 3: Linux, 32-bit, small app
    cpu, ram = predictor.predict("Linux", "32-bit", 0.8)
    print(f"Predicted for Linux, 32-bit, 0.8GB: CPU={cpu} cores, RAM={ram} GB")

    # Example 4: Linux, 64-bit, medium app
    cpu, ram = predictor.predict("Linux", "64-bit", 2.0)
    print(f"Predicted for Linux, 64-bit, 2.0GB: CPU={cpu} cores, RAM={ram} GB")
