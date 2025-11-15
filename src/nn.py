import numpy as np

class FullyConnectedNN:
    def __init__(self, layers, reg_strength=0.01, loss='softmax', seed=42, dropout_rate=0.0):
        """
        Fully connected neural network with optional dropout and L2 regularization.

        Args:
            layers: List where each element represents the number of nodes in that layer.
                Example: [input_dim, 512, 256, 8]
            reg_strength: L2 regularization strength (lambda).
                Used to penalize large weights and reduce overfitting, so the model
                prefers simpler solutions and less extreme parameter values.
            loss: 'softmax' (cross-entropy) or 'hinge' (SVM-style).
            seed: Random seed for reproducibility (weights and dropout masks).
            dropout_rate: Probability of dropping a neuron in hidden layers during training.
                Example: 0.2 means each hidden neuron has a 20% chance to be set to 0.
                in a given forward pass (training=True). 
                During evauation, dropout is disabled so that the full network is used
                and predictions are stable/deterministic. Because we use inverted dropout
                (scaling by 1/keep_prob during training), the expected activations at test 
                time already match the training distribution without needing dropout.
        """
        np.random.seed(seed)
        self.layers = layers
        self.reg_strength = reg_strength
        self.loss_type = loss
        self.dropout_rate = dropout_rate
        self.params = self._initialize_weights(layers)

    def _initialize_weights(self, layers):
        """
        Initialize weights and biases for each layer.

        For layer i (1..L-1):
            W_i has shape (layers[i-1], layers[i])
            b_i has shape (1, layers[i])

        We use He initialization (Kaiming He et al.) scaled by sqrt(2 / fan_in) 
        for better stability with ReLU / leaky-ReLU activations.

        Where:
            L = total number of layers
            fan_in = layers[i-1] (number of input connections to each neuron in layer i)
        """
        
        params = {}
        for i in range(1, len(layers)):
            # Example with a single hidden layer:
            #     W1: (2352, 500) where layers[0] is number of pixels, layers[1] number of hidden units
            #     b1: (1, 500) 
            #     W2: (500, 8) 
            #     b2: (1, 8) 
            #     np.sqrt(2.0 / layers[i-1]) is the He-init scaling factor.
            params['W' + str(i)] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2.0 / layers[i-1])
            params['b' + str(i)] = np.zeros((1, layers[i]))
        return params

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def leaky_relu(self, Z, alpha=0.01):
        return np.where(Z > 0, Z, alpha * Z)
    
    def leaky_relu_derivative(self, Z, alpha=0.01):
        return np.where(Z > 0, 1.0, alpha)


    def softmax(self, Z):
        """
        Compute row-wise softmax.
        
        Subtracts the maximum value per row for numerical stability 
        (to avoid exploding exponentials).
        The output is a matrix of probabilities where each row sums to 1.

        Args:
            Z: 2D array of raw scores (logits), shape (m, C),
                where m is the batch size and C is the number of classes.
                Each row Z[i, :] contains the class scores for sample i.
        Returns:
            2D array of probabilities with the same shape (m, C)
        """
    
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def softmax_loss(self, A, y):
        '''
        Cross-entropy loss for softmax outputs.

        Args:
            A: Raw scores or logits, shape (m, C)
            y: True labels (integer class indices), shape (m,)

        Returns:
            Scalar loss = average negative log-likelihood of the correct classes.
      '''
        m = y.shape[0]
        # Convert scores to probabilities
        p = self.softmax(A)
        # Pick the probabilities of the correct classes: p[range(m), y]
        # Small probabilities will get large penalty.
        log_likelihood = -np.log(p[range(m), y] + 1e-12)
        # Average of all losses in the batch.
        loss = np.sum(log_likelihood) / m
        return loss

    def hinge_loss(self, A, y):
        """
        Multiclass hinge loss (SVM-style) on raw scores A.

        Args:
            A: Raw scores, shape (m, C)
            y: True labels, shape (m,)

        Returns:
            Scalar hinge loss = average margin violation across the batch.
        """
        m = y.shape[0]
        correct_class_scores = A[range(m), y].reshape(-1, 1)
        margins = np.maximum(0, A - correct_class_scores + 1)
        margins[range(m), y] = 0 # no margin for correct class
        loss = np.sum(margins) / m
        return loss

    def compute_loss(self, A, y):
        """
        Compute total loss = data loss (softmax/hinge) + L2 regularization.
        """
        if self.loss_type == 'softmax':
            return self.softmax_loss(A, y) + self._l2_regularization()
        elif self.loss_type == 'hinge':
            return self.hinge_loss(A, y) + self._l2_regularization()

    def _l2_regularization(self):
        """
        L2 regularization term: (λ / 2) * sum(W_i^2)
        Penalizes large weights to avoid overfitting.

        We sum over all weight matrices but not biases.
        The factor 1/2 is conventional to make the derivative simpler:
            d/dW (λ/2 * W^2) = λ*W
        """
        reg_loss = 0
        for i in range(1, len(self.layers)):
            reg_loss += np.sum(np.square(self.params['W' + str(i)]))
        return self.reg_strength * reg_loss / 2

    def forward(self, X, training=False):
        """
        Forward pass through the network.

        If training=True:
            - Hidden layers use leaky-ReLU + (optional) dropout.
            - Dropout is applied by sampling a random mask per batch and per layer.
              Each hidden neuron is kept with probability (1 - dropout_rate)
              and scaled by 1/keep_prob ("inverted dropout").
              This means the expected activation stays the same, but some neurons
              are randomly set to zero, which forces the network not to rely
              on a single "winner" neuron.

        If training=False (e.g. validation/test/prediction):
            - Dropout is disabled (no units are dropped, full network is used).
        """
        cache = {'A0': X} # cache stores activations and pre-activations for backprop
        A = X
        for i in range(1, len(self.layers)):
            # W1: 2352, 500
            # b1: 1, 500
            # W2: 500, 8
            # b2: 1, 8
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            Z = np.dot(A, W) + b
            is_last = (i == len(self.layers) - 1)

            # Use leaky-ReLU for all hidden layers,
            # and keep the output layer linear (raw scores for the loss).
            if not is_last:
                A = self.leaky_relu(Z)
                # Dropout (only during training, on hidden layers)
                if training and getattr(self, 'dropout_rate', 0.0) > 0.0:
                    # keep_prob = probability of keeping a neuron active
                    keep_prob = 1.0 - self.dropout_rate
                    # Random mask: each neuron independently has keep_prob of staying on
                    mask = (np.random.rand(*A.shape) < keep_prob).astype(float)
                    # Inverted dropout: scale activations to keep expected value unchanged
                    A = (A * mask) / keep_prob
                    # Store mask so that backward pass uses the same dropped units
                    cache[f'D{i}'] = mask
            else:
                A = Z  # No activation in the output layer (raw scores for loss)
                
            # Z_i: pre-activation, A_i: post-activation (after nonlinearity & dropout)
            cache['Z' + str(i)] = Z 
            cache['A' + str(i)] = A # A for Activation
        return A, cache

    def backward(self, cache, y):
        """
        Backward pass (backpropagation).

        Goal:
            Compute gradients of the loss with respect to all weights and biases.

        Strategy:
            - Start from the output layer (using the chosen loss function).
            - Propagate gradients backwards through each layer using the chain rule.
            - For layers where dropout was used, apply the same dropout mask
              to the gradients, so dropped neurons also have zero gradient.
        """
        grads = {} 
        m = y.shape[0]  
        A_last = cache['A' + str(len(self.layers) - 1)]

        # ----- 1) Gradient at the output layer -----
        if self.loss_type == 'softmax':
            # For softmax + cross-entropy:
            # dL/dZ = (softmax(A_last) - one_hot(y)) / m
            dA = self.softmax(A_last)
            dA[range(m), y] -= 1     
            dA /= m             

        elif self.loss_type == 'hinge':
            # For hinge loss:
            # if margin > 0 → contributes 1, else 0 (per class)
            margins = (A_last - A_last[range(m), y].reshape(-1, 1) + 1) > 0
            margins[range(m), y] = 0          # no penalty for the correct class
            dA = np.where(margins, 1.0, 0.0)  
            row_sum = np.sum(dA, axis=1)            
            dA[np.arange(m), y] = -row_sum      # correct class gets negative sum of active margins         
            dA /= m  

        # ----- 2) Loop backwards through layers -----
        for i in reversed(range(1, len(self.layers))):
            dZ = dA  # derivative of loss w.r.t. Z at layer i
            A_prev = cache['A' + str(i - 1)]  # activation from previous layer

            # Gradients w.r.t. weights and biases:
            # dW_i = A_(i-1)^T @ dZ + lambda * W_i  (L2 regularization)
            grads['W' + str(i)] = np.dot(A_prev.T, dZ) + self.reg_strength * self.params['W' + str(i)]

            # db_i = sum over batch of dZ
            grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True)

            # ----- 3) Propagate gradient to previous layer (if not input layer) -----
            if i > 1:
                # Chain rule:
                # dA_prev = (dZ @ W_i^T) * f'(Z_(i-1))
                dA = np.dot(dZ, self.params['W' + str(i)].T) * self.leaky_relu_derivative(cache['Z' + str(i - 1)])

                # If dropout was used in layer (i-1), apply the same mask here:
                # neurons that were dropped in forward must also have zero gradient.
                if f'D{i-1}' in cache:
                    keep_prob = 1.0 - self.dropout_rate
                    dA = (dA * cache[f'D{i-1}']) / keep_prob

        # grads now holds dW_i and db_i for all layers
        return grads

    def update_params(self, grads, learning_rate, v=None, beta1=0.9, beta2=0.999, t=1, optimizer='sgd'):
        """
        Update parameters using the chosen optimization method.

        Args:
            grads: Dictionary with gradients for each W_i and b_i.
            learning_rate: Base learning rate.
            v: Velocity/moment estimates for momentum/Adam.
            beta1, beta2: Exponential decay rates for Adam.
            t: Time step (used in Adam for bias correction).
            optimizer: 'sgd', 'momentum', or 'adam'.
        """
        # Update weights depending on choice of optimizer:
        #  - 'sgd'      : plain gradient descent
        #  - 'momentum' : gradient descent with velocity (inertia)
        #  - 'adam'     : adaptive method combining momentum + per-parameter learning rates
        for i in range(1, len(self.layers)):
            if optimizer == 'sgd':
                self.params['W' + str(i)] -= learning_rate * grads['W' + str(i)]
                self.params['b' + str(i)] -= learning_rate * grads['b' + str(i)]
            elif optimizer == 'momentum':
                v['dW' + str(i)] = beta1 * v['dW' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['db' + str(i)] = beta1 * v['db' + str(i)] + (1 - beta1) * grads['b' + str(i)]
                self.params['W' + str(i)] -= learning_rate * v['dW' + str(i)]
                self.params['b' + str(i)] -= learning_rate * v['db' + str(i)]
            elif optimizer == 'adam':
                # Weights: first and second moment estimates
                v['mW' + str(i)] = beta1 * v['mW' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['vW' + str(i)] = beta2 * v['vW' + str(i)] + (1 - beta2) * np.square(grads['W' + str(i)])
                mW_hat = v['mW' + str(i)] / (1 - beta1**t)
                vW_hat = v['vW' + str(i)] / (1 - beta2**t)
                self.params['W' + str(i)] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + 1e-8)

                # Biases: same Adam update, separate moments
                v['mb' + str(i)] = beta1 * v['mb' + str(i)] + (1 - beta1) * grads['b' + str(i)]
                v['vb' + str(i)] = beta2 * v['vb' + str(i)] + (1 - beta2) * np.square(grads['b' + str(i)])
                mb_hat = v['mb' + str(i)] / (1 - beta1**t)
                vb_hat = v['vb' + str(i)] / (1 - beta2**t)
                self.params['b' + str(i)] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=64, learning_rate=0.01, optimizer='sgd', lr_decay_steps=None, lr_decay_factor=None):
        """
        Train the network for a given number of epochs using mini-batch gradient descent.

        Training loop:
            For each epoch:
                - Shuffle the training data
                - Split into mini-batches
                - For each batch:
                    Forward (training=True) → Loss → Backward → Parameter update
                - Optionally evaluate on validation set (no dropout) and log metrics.
                - Optionally apply simple step-based learning rate decay.
        """
        # Initialize velocity/moment for momentum/Adam, if needed
        v = None
        if optimizer in ['momentum', 'adam']:
            v = {}
            for i in range(1, len(self.layers)):
                v['dW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                v['db' + str(i)] = np.zeros_like(self.params['b' + str(i)])
                if optimizer == 'adam':
                    v['mW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                    v['vW' + str(i)] = np.zeros_like(self.params['W' + str(i)])
                    v['mb' + str(i)] = np.zeros_like(self.params['b' + str(i)])  # NY
                    v['vb' + str(i)] = np.zeros_like(self.params['b' + str(i)])  # NY

        # History for plotting and analysis
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        # Epoch loop (one pass over the training set)
        t = 0 # time step for Adam bias correction
        for epoch in range(epochs):
            # Shuffle data at the beginning of each epoch            
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # Mini-batch loop
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass (training=True enables dropout in hidden layers)
                A_last, cache = self.forward(X_batch, training=True)
                loss = self.compute_loss(A_last, y_batch)

                # Backward pass
                grads = self.backward(cache, y_batch)
                
                # Increase time step for Adam
                t += 1

                # Update weights with chosen optimizer
                self.update_params(grads, learning_rate, v=v, optimizer=optimizer, t=t)
              
            history['train_loss'].append(loss)
            
            # Optional validation evaluation (dropout disabled)
            if X_val is not None and y_val is not None:
                A_val, _ = self.forward(X_val, training=False)
                val_loss = self.compute_loss(A_val, y_val)
                history['val_loss'].append(val_loss)
                val_acc = (np.argmax(A_val, axis=1) == y_val).mean()
                history['val_acc'].append(val_acc)

            # Simple step learning rate decay
            if lr_decay_steps is not None and lr_decay_factor is not None:
                if (epoch + 1) % lr_decay_steps == 0:
                    learning_rate *= lr_decay_factor
            
            # Status print for first and last epoch
            if epoch in (0, epochs - 1):
                if X_val is not None:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}, Val loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")

            # # Statusprint for every 10. epoch
            # if epoch % 10 == 0:
            #     if X_val is not None:
            #         print(f"Epoch {epoch}, Loss: {loss:.4f}, Val loss: {val_loss:.4f}")
            #     else:
            #         print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return history

    def predict(self, X):
        """
        Predict class labels for input X.

        Uses a forward pass with training=False (no dropout) and picks argmax over scores.
        """        
        A_last, _ = self.forward(X)
        return np.argmax(A_last, axis=1)

