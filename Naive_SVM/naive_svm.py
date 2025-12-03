import numpy as np
import matplotlib.pyplot as plt

LR = 0.01
LP = 0.01
ITERS = 10_000

class NaiveSVM:
    def __init__(self, learning_rate, lambda_param, num_iters):
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.lambda_param = lambda_param    # Regularization parameter
        self.num_iters = num_iters          # Number of iterations
        self.w = None                       # Weights
        self.b = 0                          # Bias term

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the SVM using gradient descent."""
        num_samples, num_features = X_train.shape
        self.w = np.zeros(num_features)

        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        for i in range(self.num_iters):
            # Calculate margin
            margin = y_train * (np.dot(X_train, self.w) + self.b)

            # Compute gradient for weights and bias
            dw = self.lambda_param * self.w - (np.mean((margin < 1)[:, None] * y_train[:, None] * X_train, axis=0))
            db = -np.mean((margin < 1) * y_train)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            train_loss = self._compute_loss(X_train, y_train)
            train_accuracy = self._compute_accuracy(X_train, y_train)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)

            try:
                val_loss = self._compute_loss(X_val, y_val)
                val_accuracy = self._compute_accuracy(X_val, y_val)
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_accuracy)
            except:
                pass

        return train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def _compute_loss(self, X, y):
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b))
        return np.mean(hinge_loss) + 0.5 * self.lambda_param * np.sum(self.w ** 2)

    def _compute_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

"""
Split the dataset into training and validation sets.
"""
def split_dataset(X, y, val_ratio=0.2):
    num_samples = X.shape[0]
    val_size = int(num_samples * val_ratio)
    indices = np.random.permutation(num_samples)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

def main():
    dataset_path = 'Naive_SVM/mmwave_dataset.npy'
    data = np.load(dataset_path, allow_pickle=True).item()
    X = data['features']
    y = data['labels']

    # Preprocess labels for binary class
    unique_labels = np.unique(y)
    mid_index = len(unique_labels) // 2
    y_binary = np.where(y < unique_labels[mid_index], 1, -1)
    X_train, y_train, X_val, y_val = split_dataset(X, y_binary)

    # Train the SVM
    svm = NaiveSVM(learning_rate=LR, lambda_param=LP, num_iters=ITERS)
    train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = svm.fit(X_train, y_train, X_val, y_val)

    # Plot loss
    plt.plot(train_loss_history, label="Training Loss", color="blue")
    plt.plot(val_loss_history, label="Validation Loss", color="orange")
    plt.title("Loss")
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(train_accuracy_history, label="Training Accuracy", color="green")
    plt.plot(val_accuracy_history, label="Validation Accuracy", color="red")
    plt.title("Accuracy")
    plt.xlabel("Iters")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
