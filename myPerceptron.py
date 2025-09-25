import numpy as np

class myPerceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        # add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        # initialize weights
        self.w = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w += update * xi
        return self

    def net_input(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ---------------------------------------------------
# Test code
# ---------------------------------------------------
if __name__ == "__main__":
    # Linearly separable dataset (AND gate)
    X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([-1,-1,-1,1])  # only [1,1] is positive

    model = myPerceptron(lr=0.1, epochs=10)
    model.fit(X_and, y_and)

    print("=== AND gate (linearly separable) ===")
    print("Weights:", model.w)
    print("Predictions:", model.predict(np.c_[np.ones(X_and.shape[0]), X_and]))

    # Non-linearly separable dataset (XOR gate)
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_xor = np.array([-1,1,1,-1])  # XOR pattern

    model.fit(X_xor, y_xor)

    print("\n=== XOR gate (not linearly separable) ===")
    print("Weights:", model.w)
    print("Predictions:", model.predict(np.c_[np.ones(X_xor.shape[0]), X_xor]))
