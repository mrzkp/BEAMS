import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
import pickle

"""
MODEL ARCH
Iterative SMO Trainer for multi-class beam selection using one-vs-one SVM.
Modified to have iterative fits for monitoring progress.
Each binary classifier has its own SMO state.
In each iteration, perform one full pass over all examples for each binary classifier.
Monitor regularization loss (average 0.5 ||w||^2 over all hyperplanes) and accuracy.
"""

class MmWaveDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class BinarySMO:
    def __init__(self, C: float, tol: float):
        self.C = C
        self.tol = tol
        self.alphas = None
        self.b = 0.0
        self.K = None
        self.E = None
        self.X = None
        self.y = None
        self.is_linear_kernel = True
    
    def initialize(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y.astype(float)
        self.alphas = np.zeros(len(y))
        self.b = 0.0
        self.K = X @ X.T
        self.E = -self.y
    
    def _select_second_alpha(self, i: int, E_i: float) -> int:
        non_zero_C = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
        if len(non_zero_C) > 0:
            delta_E = np.abs(E_i - self.E[non_zero_C])
            j = non_zero_C[np.argmax(delta_E)]
            if j == i:
                j = np.random.randint(0, len(self.alphas))
                while j == i:
                    j = np.random.randint(0, len(self.alphas))
            return j
        else:
            j = np.random.randint(0, len(self.alphas))
            while j == i:
                j = np.random.randint(0, len(self.alphas))
            return j
    
    def step(self, i: int) -> int:
        E_i = np.dot(self.alphas * self.y, self.K[i, :]) + self.b - self.y[i]
        if ((self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or
            (self.y[i] * E_i > self.tol and self.alphas[i] > 0)):
            j = self._select_second_alpha(i, E_i)
            E_j = np.dot(self.alphas * self.y, self.K[j, :]) + self.b - self.y[j]
            alpha_i_old = self.alphas[i]
            alpha_j_old = self.alphas[j]
            if self.y[i] != self.y[j]:
                L = max(0, alpha_j_old - alpha_i_old)
                H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            else:
                L = max(0, alpha_i_old + alpha_j_old - self.C)
                H = min(self.C, alpha_i_old + alpha_j_old)
            if L == H:
                return 0
            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0
            self.alphas[j] = alpha_j_old - self.y[j] * (E_i - E_j) / eta
            self.alphas[j] = np.clip(self.alphas[j], L, H)
            if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                return 0
            self.alphas[i] = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])
            b1 = self.b - E_i - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i] - \
                 self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j]
            b2 = self.b - E_j - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j] - \
                 self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j]
            if 0 < self.alphas[i] < self.C:
                self.b = b1
            elif 0 < self.alphas[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2
            delta_alpha_i = self.alphas[i] - alpha_i_old
            delta_alpha_j = self.alphas[j] - alpha_j_old
            self.E += self.y[i] * delta_alpha_i * self.K[:, i] + self.y[j] * delta_alpha_j * self.K[:, j]
            return 1
        return 0

class IterativeSMOTrainer:
    def __init__(self, C: float = 1.0, tol: float = 1e-3, max_iter: int = 5):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.binary_classifiers = {}
    
    def train(self, X: np.ndarray, y: np.ndarray, NC: int):
        self.binary_classifiers = {}
        for m in range(NC):
            for n in range(m + 1, NC):
                mask = np.logical_or(y == m, y == n)
                X_sub = X[mask]
                y_sub = np.where(y[mask] == m, 1, -1)
                clf = BinarySMO(self.C, self.tol)
                clf.initialize(X_sub, y_sub)
                self.binary_classifiers[(m, n)] = clf
    
    def fit(self):
        for key, clf in self.binary_classifiers.items():
            num_samples = len(clf.y)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in indices:
                clf.step(i)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.binary_classifiers:
            raise ValueError("Model not trained yet")
        num_samples = X.shape[0]
        num_classes = max(max(k) for k in self.binary_classifiers.keys()) + 1
        votes = np.zeros((num_samples, num_classes))
        for (m, n), clf in self.binary_classifiers.items():
            sv = clf.X
            sv_y = clf.y
            alphas = clf.alphas
            b = clf.b
            K_test = X @ sv.T
            decision = (K_test @ (alphas * sv_y)) + b
            votes[decision >= 0, m] += 1
            votes[decision < 0, n] += 1
        return np.argmax(votes, axis=1)

def get_reg_loss(model):
    loss = 0.0
    for clf in model.binary_classifiers.values():
        w = np.dot(clf.alphas * clf.y, clf.X)
        loss += 0.5 * np.sum(w ** 2)
    if len(model.binary_classifiers) > 0:
        loss /= len(model.binary_classifiers)
    return loss

def train_iteration(model, X_train, y_train):
    model.fit()
    loss = get_reg_loss(model)
    y_pred = model.predict(X_train)
    acc = 100. * np.mean(y_pred == y_train)
    return loss, acc

def evaluate_model(model, X, y):
    loss = get_reg_loss(model)
    y_pred = model.predict(X)
    acc = 100. * np.mean(y_pred == y)
    return loss, acc, y_pred, y

def plot_training_history(train_losses, train_acc, val_losses, val_acc, save_path='training_history_advanced.png'):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    ax1.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(train_acc, label='Training Accuracy', color='blue', alpha=0.7)
    ax2.plot(val_acc, label='Validation Accuracy', color='red', alpha=0.7)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    X = np.load('NN_code/mmwave_features.npy')
    y = np.load('NN_code/mmwave_labels.npy')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dataset = MmWaveDataset(X_scaled, y)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    X_train = np.array([x for x, _ in train_dataset])
    y_train = np.array([y for _, y in train_dataset])
    X_val = np.array([x for x, _ in val_dataset])
    y_val = np.array([y for _, y in val_dataset])
    X_test = np.array([x for x, _ in test_dataset])
    y_test = np.array([y for _, y in test_dataset])
    model = IterativeSMOTrainer(C=1.0, tol=1e-3, max_iter=5)
    NC = len(np.unique(y))
    model.train(X_train, y_train, NC)
    num_epochs = model.max_iter
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    print("Training the SVM...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_iteration(model, X_train, y_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_loss, val_acc, _, _ = evaluate_model(model, X_val, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            with open('best_mmwave_model_advanced.pkl', 'wb') as f:
                pickle.dump(model, f)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    with open('best_mmwave_model_advanced.pkl', 'rb') as f:
        model = pickle.load(f)
    _, test_acc, _, _ = evaluate_model(model, X_test, y_test)
    print(f'\nBest Model Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()