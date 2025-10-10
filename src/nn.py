import numpy as np

class FullyConnectedNN:
    def __init__(self, layers, reg_strength=0.0, loss='softmax', seed=42, dropout_rate=0.0):
        np.random.seed(seed)
        self.layers = layers
        self.reg_strength = reg_strength
        self.loss_type = loss
        self.dropout_rate = dropout_rate
        self.params = self._init_weights(layers)

    def _init_weights(self, layers):
        p = {}
        for i in range(1, len(layers)):
            p[f"W{i}"] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2.0 / layers[i-1])  # He-init
            p[f"b{i}"] = np.zeros((1, layers[i]))
        return p

    # --- activations ---
    def leaky_relu(self, Z, alpha=0.01): return np.where(Z > 0, Z, alpha * Z)
    def relu_derivative(self, Z): return (Z > 0).astype(float)

    # --- losses ---
    def softmax(self, Z):
        Zs = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Zs)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def softmax_loss(self, A, y):
        m = y.shape[0]
        p = self.softmax(A)
        loss = -np.mean(np.log(p[np.arange(m), y] + 1e-12))
        return loss

    def hinge_loss(self, A, y):
        m = y.shape[0]
        correct = A[np.arange(m), y].reshape(-1,1)
        margins = np.maximum(0, A - correct + 1.0)
        margins[np.arange(m), y] = 0
        return np.sum(margins) / m

    def _l2(self):
        reg = 0.0
        for i in range(1, len(self.layers)):
            reg += np.sum(self.params[f"W{i}"]**2)
        return 0.5 * self.reg_strength * reg

    def compute_loss(self, A, y):
        base = self.softmax_loss(A, y) if self.loss_type == 'softmax' else self.hinge_loss(A, y)
        return base + self._l2()

    # --- forward/backward ---
    def forward(self, X, training=False):
        cache = {"A0": X}
        A = X
        for i in range(1, len(self.layers)):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]
            Z = A @ W + b
            last = (i == len(self.layers) - 1)
            if not last:
                A = self.leaky_relu(Z)
                if training and self.dropout_rate > 0.0:
                    keep = 1.0 - self.dropout_rate
                    mask = (np.random.rand(*A.shape) < keep).astype(float)
                    A = (A * mask) / keep
                    cache[f"D{i}"] = mask
            else:
                A = Z  # logits
            cache[f"Z{i}"], cache[f"A{i}"] = Z, A
        return A, cache

    def backward(self, cache, y):
        grads = {}
        m = y.shape[0]
        A_last = cache[f"A{len(self.layers)-1}"]

        if self.loss_type == 'softmax':
            dA = self.softmax(A_last)
            dA[np.arange(m), y] -= 1
            dA /= m
        else:
            margins = (A_last - A_last[np.arange(m), y].reshape(-1,1) + 1.0) > 0
            margins[np.arange(m), y] = 0
            dA = margins.astype(float)
            row_sum = np.sum(dA, axis=1)
            dA[np.arange(m), y] = -row_sum
            dA /= m

        for i in reversed(range(1, len(self.layers))):
            dZ = dA
            A_prev = cache[f"A{i-1}"]
            grads[f"W{i}"] = A_prev.T @ dZ + self.reg_strength * self.params[f"W{i}"]
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True)

            if i > 1:
                dA = dZ @ self.params[f"W{i}"].T * self.relu_derivative(cache[f"Z{i-1}"])
                if f"D{i-1}" in cache:
                    keep = 1.0 - self.dropout_rate
                    dA = (dA * cache[f"D{i-1}"]) / keep
        return grads

    # --- update ---
    def update_params(self, grads, lr, v=None, beta1=0.9, beta2=0.999, t=1, optimizer='adam'):
        for i in range(1, len(self.layers)):
            if optimizer == 'sgd':
                self.params[f"W{i}"] -= lr * grads[f"W{i}"]
                self.params[f"b{i}"] -= lr * grads[f"b{i}"]
            elif optimizer == 'momentum':
                v[f"dW{i}"] = beta1*v[f"dW{i}"] + (1-beta1)*grads[f"W{i}"]
                v[f"db{i}"] = beta1*v[f"db{i}"] + (1-beta1)*grads[f"b{i}"]
                self.params[f"W{i}"] -= lr * v[f"dW{i}"]
                self.params[f"b{i}"] -= lr * v[f"db{i}"]
            else:  # adam
                v[f"mW{i}"] = beta1*v[f"mW{i}"] + (1-beta1)*grads[f"W{i}"]
                v[f"vW{i}"] = beta2*v[f"vW{i}"] + (1-beta2)*(grads[f"W{i}"]**2)
                mW_hat = v[f"mW{i}"] / (1 - beta1**t)
                vW_hat = v[f"vW{i}"] / (1 - beta2**t)
                self.params[f"W{i}"] -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)

                v[f"mb{i}"] = beta1*v[f"mb{i}"] + (1-beta1)*grads[f"b{i}"]
                v[f"vb{i}"] = beta2*v[f"vb{i}"] + (1-beta2)*(grads[f"b{i}"]**2)
                mb_hat = v[f"mb{i}"] / (1 - beta1**t)
                vb_hat = v[f"vb{i}"] / (1 - beta2**t)
                self.params[f"b{i}"] -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # --- fit/predict ---
    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=128,
            learning_rate=1e-3, optimizer='adam', lr_decay_steps=None, lr_decay_factor=None):
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        v = None
        if optimizer in ['momentum', 'adam']:
            v = {}
            for i in range(1, len(self.layers)):
                v[f"dW{i}"] = np.zeros_like(self.params[f"W{i}"])
                v[f"db{i}"] = np.zeros_like(self.params[f"b{i}"])
                if optimizer == 'adam':
                    v[f"mW{i}"] = np.zeros_like(self.params[f"W{i}"])
                    v[f"vW{i}"] = np.zeros_like(self.params[f"W{i}"])
                    v[f"mb{i}"] = np.zeros_like(self.params[f"b{i}"])
                    v[f"vb{i}"] = np.zeros_like(self.params[f"b{i}"])

        t = 0
        for epoch in range(epochs):
            perm = np.random.permutation(X.shape[0])
            Xs, ys = X[perm], y[perm]
            for i in range(0, X.shape[0], batch_size):
                Xb, yb = Xs[i:i+batch_size], ys[i:i+batch_size]
                logits, cache = self.forward(Xb, training=True)
                loss = self.compute_loss(logits, yb)
                grads = self.backward(cache, yb)
                t += 1
                self.update_params(grads, learning_rate, v=v, optimizer=optimizer, t=t)

            history['train_loss'].append(loss)
            if X_val is not None and y_val is not None:
                logits_val, _ = self.forward(X_val, training=False)
                vloss = self.compute_loss(logits_val, y_val)
                vacc = (np.argmax(logits_val, axis=1) == y_val).mean()
                history['val_loss'].append(vloss)
                history['val_acc'].append(vacc)

            if lr_decay_steps and lr_decay_factor and (epoch+1) % lr_decay_steps == 0:
                learning_rate *= lr_decay_factor

        return history

    def predict(self, X):
        logits, _ = self.forward(X, training=False)
        return np.argmax(logits, axis=1)
