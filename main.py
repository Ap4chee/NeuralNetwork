import numpy as np
from PIL import Image
import os

LABELS = ['jednolity', 'poziomy', 'pionowy', 'skosny']


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(np.clip(z_shifted, -500, 500))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def generate_realistic_dataset(samples_per_class=500):
    X, y = [], []

    for cls in range(4):
        for i in range(samples_per_class):
            if cls == 0:
                base_brightness = np.random.uniform(0.1, 0.9)
                pattern = np.array([base_brightness] * 4) + np.random.normal(0, 0.05, 4)

            elif cls == 1:
                top_brightness = np.random.uniform(0.1, 0.9)
                bottom_brightness = np.random.uniform(0.1, 0.9)
                if abs(top_brightness - bottom_brightness) < 0.2:
                    bottom_brightness = top_brightness + 0.3 if top_brightness < 0.6 else top_brightness - 0.3
                pattern = np.array([top_brightness, top_brightness, bottom_brightness, bottom_brightness])
                pattern += np.random.normal(0, 0.03, 4)

            elif cls == 2:
                left_brightness = np.random.uniform(0.1, 0.9)
                right_brightness = np.random.uniform(0.1, 0.9)
                if abs(left_brightness - right_brightness) < 0.2:
                    right_brightness = left_brightness + 0.3 if left_brightness < 0.6 else left_brightness - 0.3
                pattern = np.array([left_brightness, right_brightness, left_brightness, right_brightness])
                pattern += np.random.normal(0, 0.03, 4)

            else:
                diag1_brightness = np.random.uniform(0.1, 0.9)  # piksele 0,3
                diag2_brightness = np.random.uniform(0.1, 0.9)  # piksele 1,2
                if abs(diag1_brightness - diag2_brightness) < 0.2:
                    diag2_brightness = diag1_brightness + 0.3 if diag1_brightness < 0.6 else diag1_brightness - 0.3
                pattern = np.array([diag1_brightness, diag2_brightness, diag2_brightness, diag1_brightness])
                pattern += np.random.normal(0, 0.03, 4)

            pattern = np.clip(pattern, 0.0, 1.0)
            X.append(pattern)
            y.append(cls)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)

    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    for cls in range(4):
        sample = X[y == cls][0]
        print(f"  {LABELS[cls]}: [{sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}, {sample[3]:.3f}]")

    for cls in range(4):
        class_data = X[y == cls]
        mean_vals = np.mean(class_data, axis=0)
        print(
            f"  {LABELS[cls]} - średnie: [{mean_vals[0]:.3f}, {mean_vals[1]:.3f}, {mean_vals[2]:.3f}, {mean_vals[3]:.3f}]")

    return X, y


class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=8, output_size=4):
        self.W1 = np.random.normal(0, np.sqrt(2.0 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def compute_loss(self, predictions, Y):
        m = Y.shape[0]
        predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        log_likelihood = -np.log(predictions_clipped[range(m), Y])
        return np.mean(log_likelihood)

    def backward(self, X, Y, lr):
        m = X.shape[0]

        Y_onehot = np.zeros((m, 4))
        Y_onehot[range(m), Y] = 1

        dZ2 = (self.A2 - Y_onehot) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grad_norm = np.sqrt(np.sum(dW1 ** 2) + np.sum(dW2 ** 2))
        if grad_norm > 1.0:
            dW1 = dW1 / grad_norm
            dW2 = dW2 / grad_norm
            db1 = db1 / grad_norm
            db2 = db2 / grad_norm

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return grad_norm

    def train(self, X, Y, epochs, lr):
        best_accuracy = 0
        patience = 100
        no_improvement = 0

        for epoch in range(1, epochs + 1):
            predictions = self.forward(X)
            loss = self.compute_loss(predictions, Y)

            grad_norm = self.backward(X, Y, lr)

            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == Y)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improvement = 0
            else:
                no_improvement += 1

            if epoch % max(epochs // 20, 1) == 0 or epoch <= 5:
                print(f"Epoka {epoch:4d}/{epochs}: loss={loss:.6f}, accuracy={accuracy:.4f}, grad_norm={grad_norm:.6f}")

                if epoch % (epochs // 5) == 0:
                    self.print_confusion_matrix(Y, pred_classes)

            if no_improvement > patience and accuracy > 0.95:
                print(f"Early stopping na epoce {epoch}, najlepsza accuracy: {best_accuracy:.4f}")
                break

    def print_confusion_matrix(self, y_true, y_pred):
        print("  Confusion Matrix:")
        for true_cls in range(4):
            row = []
            for pred_cls in range(4):
                count = np.sum((y_true == true_cls) & (y_pred == pred_cls))
                row.append(f"{count:4d}")
            print(f"    {LABELS[true_cls]:8s}: [" + " ".join(row) + "]")

    def predict_proba(self, X):
        return self.forward(X.reshape(1, -1))[0]

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba), proba


def load_and_preprocess_image(path):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((2, 2), Image.LANCZOS)
        pixels = np.array(img).reshape(-1, 3)
        features = []
        for r, g, b in pixels:
            brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            features.append(brightness)
        features = np.array(features, dtype=np.float32)
        return features

    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        raise


def analyze_pattern(features):
    print(f"\n=== ANALIZA WZORCA ===")

    pattern = features.reshape(2, 2)
    print("Rozkład jasności (2x2):")
    print(f"  {pattern[0, 0]:.3f}  {pattern[0, 1]:.3f}")
    print(f"  {pattern[1, 0]:.3f}  {pattern[1, 1]:.3f}")


def save_weights(model, path):
    np.savez(path, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print(f"Zapisano wagi do {path}")


def load_weights(model, path):
    data = np.load(path)
    model.W1, model.b1 = data['W1'], data['b1']
    model.W2, model.b2 = data['W2'], data['b2']
    print(f"Wczytano wagi z {path}")


def main():
    print("Wybierz cyfre:")
    print("1 - Trening sieci na realistycznych danych")
    print("2 - Test na obrazie")
    print("3 - Test na kilku obrazach (batch)")
    mode = input("Podaj 1, 2 lub 3: ").strip()

    model = NeuralNetwork(input_size=4, hidden_size=8, output_size=4)
    weights_path = 'weights.npz'

    if mode == '1':
        print("\nTRENING SIECI")
        samples = int(input("Próbki na klasę [1000]: ") or '1000')
        epochs = int(input("Liczba epok [2000]: ") or '2000')
        lr = float(input("Learning rate [1]: ") or '1')

        X, y = generate_realistic_dataset(samples)

        model.train(X, y, epochs, lr)

        save_weights(model, weights_path)

        print("\nTrening zakończony!")

    elif mode == '2':
        print("\nTEST NA OBRAZIE")
        if not os.path.exists(weights_path):
            print("Błąd: Brak wytrenowanych wag! Wytrenuj model najpierw (opcja 1).")
            return

        img_path = input("Ścieżka do obrazu: ").strip()
        if not os.path.isfile(img_path):
            print("Błąd: Plik nie istnieje.")
            return

        load_weights(model, weights_path)

        try:
            features = load_and_preprocess_image(img_path)

            analyze_pattern(features)

            cls, proba = model.predict(features)

            print(f"\nWYNIK KLASYFIKACJI: ")
            print(f"Obraz sklasyfikowano jako: {LABELS[cls]} (klasa {cls})")
            print("Prawdopodobieństwa:")
            for i, p in enumerate(proba):
                marker = " <-- WYBRANE" if i == cls else ""
                print(f"  {LABELS[i]}: {p * 100:.2f}%{marker}")

        except Exception as e:
            print(f"Błąd podczas przetwarzania obrazu: {e}")

    elif mode == '3':
        if not os.path.exists(weights_path):
            print("Błąd: Brak wytrenowanych wag! Wytrenuj model najpierw (opcja 1).")
            return

        folder_path = input("Ścieżka do folderu z obrazami: ").strip()
        if not os.path.isdir(folder_path):
            print("Błąd: Folder nie istnieje.")
            return

        load_weights(model, weights_path)

        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])

        if not image_files:
            print("Nie znaleziono obrazów w folderze.")
            return

        print(f"Znaleziono {len(image_files)} obrazów.")

        for img_file in sorted(image_files):
            img_path = os.path.join(folder_path, img_file)
            print(f"{img_file}")

            try:
                features = load_and_preprocess_image(img_path)
                cls, proba = model.predict(features)

                print(f"Klasyfikacja: {LABELS[cls]} ({proba[cls] * 100:.1f}%)")

            except Exception as e:
                print(f"Błąd: {e}")

    else:
        print("Nieprawidłowy wybór. Wybierz 1, 2 lub 3.")


if __name__ == '__main__':
    main()
