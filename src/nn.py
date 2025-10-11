import numpy as np

class FullyConnectedNN:
    def __init__(self, layers, reg_strength=0.01, loss='softmax', seed=42, dropout_rate=0.0):
        """
        layers: List where each element represents the number of nodes in that layer.
        reg_strength: L2 regularization strength
        loss: 'softmax' or 'hinge'
        """
        np.random.seed(seed)
        self.layers = layers
        self.reg_strength = reg_strength
        self.loss_type = loss
        self.dropout_rate = dropout_rate
        self.params = self._initialize_weights(layers)

    def _initialize_weights(self, layers):
        """
        Initialize weights and biases for each layer
        """
        params = {}
        for i in range(1, len(layers)):
            # builds these matrices:
            # W1: (2352, 500) eller layers[0] antal pixels, layers[1] antal neuroner
            # - alle med random normalfordelte tal med mean = 0 og variance = 1
            # b1: (1, 500) alle initializeret med 0
            # W2: (500, 8) med random tal
            # b2: (1, 8) med 0'er
            # * np.sqrt(2.0 / layers[i-1]) er tilføjet for test med flere lag for mere stabilitet (He-init)
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
      # man trækker max fra hver enkelt af scoreværdierne i Z,
      # for at forhindre dem i at eksplodere
      # proportionerne bevares, trods denne operation
      # returnerer en række af sandsynligheder for match med klasse,
      # som tilsammen giver 1
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def softmax_loss(self, A, y):
      '''
      A = alle modellens output-scores for en batch - shape(m,C)
      y = de rigtige labels, med længden m
      m = antal prøver i batchen
      '''
      m = y.shape[0]
      # laver sandsynligheder
      p = self.softmax(A)
      # p[range(m), y] henter sandsynlighederne på de klasser der er rigtige (index y)
      # log af små tal giver stor straf,
      # for alle tal mellem 0 og 1, returnere log et negativt tal, dette gøres positivt.
      # dvs hvis den rigtige klasse ikke er sandsynlig endnu, skal den korigeres meget.
      log_likelihood = -np.log(p[range(m), y] + 1e-12)
      # gennemsnit af alle loss i batchen
      loss = np.sum(log_likelihood) / m
      return loss

    def hinge_loss(self, A, y):
      '''
      A: modellens scores (uden softmax) — form (m, C)
      (én række pr. sample, én kolonne pr. klasse)
      y: rigtige labels, længde m
      m: antal eksempler i batchen
      '''
      m = y.shape[0]
      correct_class_scores = A[range(m), y].reshape(-1, 1)
      margins = np.maximum(0, A - correct_class_scores + 1)
      margins[range(m), y] = 0
      loss = np.sum(margins) / m
      return loss

    def compute_loss(self, A, y):
        if self.loss_type == 'softmax':
            return self.softmax_loss(A, y) + self._l2_regularization()
        elif self.loss_type == 'hinge':
            return self.hinge_loss(A, y) + self._l2_regularization()

    def _l2_regularization(self):
        # L2-regularisering: straffer store vægte for at undgå overfitting
        # Alle vægte kvadreres og summeres
        # Divideres med 2 for at gøre den afledte formel (gradienten) pænere: d/dW = λW i stedet for 2λW
        reg_loss = 0
        for i in range(1, len(self.layers)):
            reg_loss += np.sum(np.square(self.params['W' + str(i)]))
        return self.reg_strength * reg_loss / 2

    def forward(self, X, training=False):
      # cache saved as dict, to be used in backward
        cache = {'A0': X}
        # A er X til at starte med og bruges sådan i dot operationen
        A = X
        for i in range(1, len(self.layers)):
          # W1: 2352, 500
          # b1: 1, 500
          # W2: 500, 8
          # b2: 1, 8
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            Z = np.dot(A, W) + b
            is_last = (i == len(self.layers) - 1)

            # Brug ReLU for alle lag undtagen det sidste —
            # det sidste lag skal bare give rå scores,
            # som loss-funktionen (softmax/hinge) arbejder på.
            # len(layers) = 3, og i starter med 1.
            # så hvis i != 2, altså kun i layer[1]
            # layer[2] er output layeret og skal indeholde negative værdier
            if not is_last:
                A = self.leaky_relu(Z)
                # Dropout (kun under træning)
                if training and getattr(self, 'dropout_rate', 0.0) > 0.0:
                    keep_prob = 1.0 - self.dropout_rate
                    mask = (np.random.rand(*A.shape) < keep_prob).astype(float)
                    A = (A * mask) / keep_prob
                    cache[f'D{i}'] = mask
            else:
                A = Z  # No activation in the output layer (raw scores for loss)

            cache['Z' + str(i)] = Z # Z2 = A2, ellers er de hhv med (A) og uden (Z) relu (activation)
            cache['A' + str(i)] = A # A for Activation
        return A, cache

    def backward(self, cache, y):
        # BACKWARD PASS
        # Formålet: Beregn hvordan loss ændrer sig, hvis vi ændrer på vægtene (gradienter)
        # Vi går baglæns gennem lagene og bruger kædereglen til at propagere fejlen tilbage.

        grads = {}  # Her gemmer vi alle gradienter (dW, db for hvert lag)
        m = y.shape[0]  # Antal eksempler i batchen
        # A_last = Output fra sidste lag (fx A2) altså outputlaget (batch_size, 8)
        A_last = cache['A' + str(len(self.layers) - 1)]

        # ----- 1️⃣ Start ved outputlaget -----
        if self.loss_type == 'softmax':
            # Softmax-loss gradient: dA = (predicted_probs - one_hot(y)) / m
            dA = self.softmax(A_last) # Konverter scores til sandsynligheder
            dA[range(m), y] -= 1      # Træk 1 fra den rigtige klasse (∂L/∂Z = p - y)
            dA /= m                   # Gennemsnit over batchen

        elif self.loss_type == 'hinge':
            # Hinge-loss gradient: hvis margin > 0 → 1, ellers 0
            margins = (A_last - A_last[range(m), y].reshape(-1, 1) + 1) > 0
            margins[range(m), y] = 0      # Ingen straf for korrekt klasse
            dA = np.where(margins, 1.0, 0.0)  # 1.0 hvis margin overtræder, 0.0 ellers (float litterals)
            row_sum = np.sum(dA, axis=1)              # antal aktive pr. sample
            dA[np.arange(m), y] = -row_sum            # korrekt klasse = -(#aktive)
            dA /= m  # Gennemsnit over batchen

        # ----- 2️⃣ Loop baglæns gennem lagene -----
        for i in reversed(range(1, len(self.layers))):
            dZ = dA  # dZ = afledt af loss ift. Z i nuværende lag
            A_prev = cache['A' + str(i - 1)]  # A_prev = output fra forrige lag (fra forward-pass)

            # Gradient for vægte og bias i dette lag
            # dW = A_prev.T @ dZ  (hvor meget hvert input påvirkede fejlen)
            # + reg_strength * W  (L2-regularisering)
            grads['W' + str(i)] = np.dot(A_prev.T, dZ) + self.reg_strength * self.params['W' + str(i)]

            # db = summen af dZ (én bias-gradient pr. neuron)
            grads['b' + str(i)] = np.sum(dZ, axis=0, keepdims=True)

            # ----- 3️⃣ Beregn dA for forrige lag (hvis der er et) -----
            if i > 1:
                # Kædereglen:
                # dA_prev = (dZ @ W.T) * ReLU'(Z_prev)
                # dvs. hvordan ændringer i dette lag påvirker det forrige
                dA = np.dot(dZ, self.params['W' + str(i)].T) * self.leaky_relu_derivative(cache['Z' + str(i - 1)])
                # Anvend dropaout-masken, hvis den var brugt i forward
                if f'D{i-1}' in cache:
                    keep_prob = 1.0 - self.dropout_rate
                    dA = (dA * cache[f'D{i-1}']) / keep_prob

        # grads indeholder nu dW og db for hvert lag, klar til vægtopdatering
        return grads

    def update_params(self, grads, learning_rate, v=None, beta1=0.9, beta2=0.999, t=1, optimizer='sgd'):
        """
        Updates parameters with chosen optimization method.
        If optimizer is 'momentum' or 'adam', it requires v for velocity and also t for time step in adam.
        """

        # Opdaterer vægte afhængigt af valgt optimizer:
        #  - 'sgd'      : simpelt skridt i negativ gradientretning
        #  - 'momentum' : tilføjer inerti (gemmer tidligere gradienter)
        #  - 'adam'     : kombinerer momentum + adaptiv læringsrate (hurtigere og stabil konvergens)
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
                # vægte
                v['mW' + str(i)] = beta1 * v['mW' + str(i)] + (1 - beta1) * grads['W' + str(i)]
                v['vW' + str(i)] = beta2 * v['vW' + str(i)] + (1 - beta2) * np.square(grads['W' + str(i)])
                mW_hat = v['mW' + str(i)] / (1 - beta1**t)
                vW_hat = v['vW' + str(i)] / (1 - beta2**t)
                self.params['W' + str(i)] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + 1e-8)
                # biases (NYT)
                v['mb' + str(i)] = beta1 * v['mb' + str(i)] + (1 - beta1) * grads['b' + str(i)]
                v['vb' + str(i)] = beta2 * v['vb' + str(i)] + (1 - beta2) * np.square(grads['b' + str(i)])
                mb_hat = v['mb' + str(i)] / (1 - beta1**t)
                vb_hat = v['vb' + str(i)] / (1 - beta2**t)
                self.params['b' + str(i)] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=64, learning_rate=0.01, optimizer='sgd', lr_decay_steps=None, lr_decay_factor=None):
        """
        Trains the model using the chosen optimizer.

        Træner modellen over flere epochs med valgt optimizer.
        Forward -> Loss -> Backward -> Opdater vægte (gentaget for alle batches).
        """

        # Initialiser velocity til momentum/adam
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

        # history til plotting
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        # Epoch loop (ét gennemløb af hele datasættet)
        t = 0 # tidsstep til adam bias-correction
        for epoch in range(epochs):
            # Shuffle data for hver epoch
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # Mini-batch loop
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Fremad, loss, bagud
                A_last, cache = self.forward(X_batch, training=True)
                loss = self.compute_loss(A_last, y_batch)
                grads = self.backward(cache, y_batch)

                t += 1 # update time for adam

                # Opdater vægte med valgte optimizer
                self.update_params(grads, learning_rate, v=v, optimizer=optimizer, t=t)
              
            history['train_loss'].append(loss)

            if X_val is not None and y_val is not None:
                A_val, _ = self.forward(X_val, training=False)
                val_loss = self.compute_loss(A_val, y_val)
                history['val_loss'].append(val_loss)
                val_acc = (np.argmax(A_val, axis=1) == y_val).mean()
                history['val_acc'].append(val_acc)

             # simpel step decay
            if lr_decay_steps is not None and lr_decay_factor is not None:
                if (epoch + 1) % lr_decay_steps == 0:
                    learning_rate *= lr_decay_factor
            
            # Statusprint ved første og sidste epoch
            if epoch in (0, epochs - 1):
                if X_val is not None:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}, Val loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")

            # # Statusprint hver 10. epoch
            # if epoch % 10 == 0:
            #     if X_val is not None:
            #         print(f"Epoch {epoch}, Loss: {loss:.4f}, Val loss: {val_loss:.4f}")
            #     else:
            #         print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return history

    def predict(self, X):
        # Beregn modelens output-scores for input X (ingen backprop her)
        A_last, _ = self.forward(X)
        # Vælg den klasse (kolonne) med højest score for hver prøve
        if self.loss_type == 'softmax':
            return np.argmax(A_last, axis=1)
        elif self.loss_type == 'hinge':
            return np.argmax(A_last, axis=1)

