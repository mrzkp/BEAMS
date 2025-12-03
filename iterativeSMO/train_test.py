import numpy as np
import matplotlib.pyplot as plt
from SMO import SVM

"""
Binary SVM training for mmWave beam classification
"""

"""
Loads mmWave dataset from npy file and preprocesses labels for binary classification.
Trains SVM model iteratively using SMO, tracking loss and accuracy over iterations.
Performance is monitored with printed metrics and visualized in a plot.
"""

"""Load the dataset from the .npy file."""
def load(filepath):
    data = np.load(filepath, allow_pickle=True).item()
    features = data['features']
    labels = data['labels']
    return features, labels

"""Convert the beam indices into a binary classification problem for SVM."""
def preprocess_labels(labels):
    unique_labels = np.unique(labels)
    mid_index = len(unique_labels) // 2
    binary_labels = np.where(labels < unique_labels[mid_index], 1, -1)
    return binary_labels

"""Calculate accuracy as the percentage of correct predictions."""
def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

def main():
    dataset_path = 'SMO_code_2/mmwave_dataset.npy'
    X, y = load(dataset_path)
    y_binary = preprocess_labels(y)

    svm = SVM(X, y_binary, C=1, kernel='linear', max_iter=300)
    loss_history = []
    accuracy_history = []
    for iteration in range(svm.max_iter):
        svm.fit()

        loss = 0.5 * np.sum(svm.w ** 2) if svm.is_linear_kernel else 0.0 # l2 loss
        loss_history.append(loss)

        predictions = np.sign([svm.predict(x) for x in X])
        accuracy = calculate_accuracy(predictions, y_binary)
        accuracy_history.append(accuracy)
        print(f"{iteration} / {svm.max_iter}, {loss:.4f}, {accuracy:.2f}")

    plt.plot(loss_history, label="Loss")
    plt.plot(accuracy_history, label="Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()