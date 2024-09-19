import numpy as np

class MLP:
    def __init__(self, layer_sizes, activations, cost_fn, optimizer, init_method, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.cost_fn = cost_fn
        self.optimizer = optimizer
        self.init_method = init_method
        self.beta1 = beta1  # Adam
        self.beta2 = beta2  # Adam
        self.epsilon = epsilon  # Adam
        self.weights, self.biases = self.initialize_weights()
        self.m = 0  # number of samples, initialized during training

        if optimizer == 'adam':
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]

    def initialize_weights(self):
        weights = []
        biases = []
        for i in range(1, len(self.layer_sizes)):
            if self.init_method == 'xavier':
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(1 / self.layer_sizes[i-1])
            elif self.init_method == 'he':
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2 / self.layer_sizes[i-1])
            else:  # normal or uniform
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1])
            b = np.zeros((self.layer_sizes[i], 1))
            weights.append(w)
            biases.append(b)
        return weights, biases
    
    def forward(self, X):
        activations = [X]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Ensure b is broadcasted correctly
            z = np.dot(w, activations[-1]) + b  # z = W * a + b
            zs.append(z)
            if self.activations[i] == 'relu':
                activations.append(self.relu(z))
            elif self.activations[i] == 'sigmoid':
                activations.append(self.sigmoid(z))
            elif self.activations[i] == 'tanh':
                activations.append(self.tanh(z))
            elif self.activations[i] == 'leaky_relu':
                activations.append(self.leaky_relu(z))
            else:  # Linear activation
                activations.append(z)
        return activations, zs
    
    def backward(self, activations, zs, y):
        weight_grads = [None] * len(self.weights)
        bias_grads = [None] * len(self.biases)
        
        if self.cost_fn == 'bce':
            dz = activations[-1] - y  # for binary cross-entropy
        elif self.cost_fn == 'mse':
            dz = (activations[-1] - y) * self.deriv_mse(activations[-1], y)
        
        for i in reversed(range(len(self.weights))):
            # dz is (units in current layer, number of samples)
            # activations[i] is (units in previous layer, number of samples)
            weight_grads[i] = np.dot(dz, activations[i].T) / self.m  # Ensure proper matrix multiplication
            bias_grads[i] = np.sum(dz, axis=1, keepdims=True) / self.m  # Sum over the sample dimension
            
            if i > 0:
                dz = np.dot(self.weights[i].T, dz) * self.deriv_activation(zs[i-1], self.activations[i-1])
        
        return weight_grads, bias_grads
    
    def train(self, X, y, epochs, batch_size, learning_rate):
        self.m = X.shape[1]  # number of samples
        for epoch in range(epochs):
            activations, zs = self.forward(X)
            loss = self.compute_cost(activations[-1], y)
            weight_grads, bias_grads = self.backward(activations, zs, y)
            self.update_parameters(weight_grads, bias_grads, learning_rate, epoch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def update_parameters(self, weight_grads, bias_grads, learning_rate, t):
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * weight_grads[i]
                self.biases[i] -= learning_rate * bias_grads[i]
        elif self.optimizer == 'adam':
            for i in range(len(self.weights)):
                # Adam updates
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_grads[i]
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_grads[i] ** 2)

                m_hat_w = self.m_weights[i] / (1 - self.beta1 ** (t + 1))
                v_hat_w = self.v_weights[i] / (1 - self.beta2 ** (t + 1))

                self.weights[i] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_grads[i]
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_grads[i] ** 2)

                m_hat_b = self.m_biases[i] / (1 - self.beta1 ** (t + 1))
                v_hat_b = self.v_biases[i] / (1 - self.beta2 ** (t + 1))

                self.biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    # Activation functions and their derivatives
    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)
    
    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    def deriv_activation(self, z, activation_fn):
        if activation_fn == 'relu':
            return np.where(z > 0, 1, 0)
        elif activation_fn == 'sigmoid':
            a = self.sigmoid(z)
            return a * (1 - a)
        elif activation_fn == 'tanh':
            return 1 - np.tanh(z)**2
        elif activation_fn == 'leaky_relu':
            return np.where(z > 0, 1, 0.01)
        else:
            return 1
        
        # Add softmax function
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Subtract max for numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    # Cost functions
    def binary_cross_entropy(self, y_pred, y_true):
        epsilon = 1e-12  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions to prevent log(0)
        m = y_true.shape[1]
        cost = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return np.squeeze(cost)
    
    def mean_squared_error(self, y_pred, y_true):
        # Ensure y_pred and y_true are numpy arrays (not numpy matrices)
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        
        return np.mean((y_pred - y_true) ** 2)
    
    def compute_cost(self, y_pred, y_true):
        if self.cost_fn == 'bce':
            return self.binary_cross_entropy(y_pred, y_true)
        elif self.cost_fn == 'mse':
            return self.mean_squared_error(y_pred, y_true)
    
    def deriv_mse(self, y_pred, y_true):
        return 2 * (y_pred - y_true)
    
    # Model export and import
    def export_model(self, filename):
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'activations': self.activations,
            'cost_fn': self.cost_fn,
            'optimizer': self.optimizer,
            'init_method': self.init_method
        }
        np.savez(filename, **model_data)

    def import_model(self, filename):
        model_data = np.load(filename, allow_pickle=True)
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']
        self.activations = model_data['activations']
        self.cost_fn = model_data['cost_fn']
        self.optimizer = model_data['optimizer']
        self.init_method = model_data['init_method']