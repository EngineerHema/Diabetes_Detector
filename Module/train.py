import numpy as np
import pandas as pd

CHOSEN_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BMI",
    "DiabetesPedigreeFunction",
    "Insulin",
    "Age",
    "Outcome"
]

LEARNING_RATE = 0.01

class Trainer:

    def __init__(self):
        df = pd.read_csv("D:\ML_pro\Diabetes_Detector\Data\diabetes_dataset.csv")
        self.data = df[CHOSEN_FEATURES]
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.W = None
        self.B = 0
        self.Y_predicted = None

        self.__prepare_data()
        self.__initialize_weights()

    def __prepare_data(self):
        # Shuffle rows
        shuffled_data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split features and target
        X = shuffled_data.drop("Outcome", axis=1).values
        Y = shuffled_data["Outcome"].values

        # 80/20 split
        split_index = int(0.8 * len(X))
        self.X_train, self.Y_train = X[:split_index], Y[:split_index]
        self.X_test, self.Y_test = X[split_index:], Y[split_index:]

        # Z-normalization (standardization) on training features
        self.mean = np.mean(self.X_train, axis=0)
        self.std = np.std(self.X_train, axis=0)
        self.X_train = (self.X_train - self.mean) / self.std

        # Apply same transformation to test features
        self.X_test = (self.X_test - self.mean) / self.std

    def __initialize_weights(self):
        n_features = self.X_train.shape[1]
        self.W = np.random.rand(n_features) * 0.001  # small random weights

    def train_with_gradient_descent(self, epochs=1000):
        for _ in range(epochs):
            # Predictions
            self.Y_predicted = np.dot(self.X_train, self.W) + self.B

            # Errors
            error = self.Y_predicted - self.Y_train

            # Gradients
            dW = (1/len(self.X_train)) * np.dot(self.X_train.T, error)
            dB = (1/len(self.X_train)) * np.sum(error)

            old_W = self.W.copy()

            # Update weights and bias
            self.W -= LEARNING_RATE * dW
            self.B -= LEARNING_RATE * dB

            # if np.linalg.norm(self.W - old_W) < 1e-40:
            #     print(f"Stopped early at iteration {_}")
            #     break


    def __predict(self, X):
        return np.dot(X, self.W) + self.B

    def evaluate(self):
        Y_pred = self.__predict(self.X_test)
        # simple accuracy for classification (threshold 0.5)
        Y_pred_class = (Y_pred >= 0.7).astype(int)
        accuracy = np.mean(Y_pred_class == self.Y_test)
        return accuracy
