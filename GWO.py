import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Load and preprocess data ---
iris = load_iris()
X, y = iris.data, iris.target.reshape(-1,1)

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Neural network structure ---
n_input = X_train.shape[1]
n_hidden = 5
n_output = y_train.shape[1]

def nn_forward(x, weights):
    """weights: flattened array containing all weights and biases"""
    # Extract weights and biases
    idx = 0
    W1 = weights[idx:idx+n_input*n_hidden].reshape(n_input, n_hidden); idx += n_input*n_hidden
    b1 = weights[idx:idx+n_hidden]; idx += n_hidden
    W2 = weights[idx:idx+n_hidden*n_output].reshape(n_hidden, n_output); idx += n_hidden*n_output
    b2 = weights[idx:idx+n_output]

    H = np.tanh(np.dot(x, W1) + b1)          # hidden layer
    out = np.dot(H, W2) + b2                 # output layer (linear)
    # softmax
    exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
    return exp_out / np.sum(exp_out, axis=1, keepdims=True)

def fitness(weights):
    y_pred = nn_forward(X_train, weights)
    # cross-entropy loss
    loss = -np.sum(y_train * np.log(y_pred+1e-8)) / y_train.shape[0]
    return loss

# --- GWO parameters ---
N = 10                   # number of wolves
MaxIter = 50
dim = n_input*n_hidden + n_hidden + n_hidden*n_output + n_output  # total weights+biases

# Initialize wolves
X = np.random.uniform(-1,1,(N, dim))
fitness_vals = np.array([fitness(xi) for xi in X])

def get_leaders(X, fitness_vals):
    idx = np.argsort(fitness_vals)
    return X[idx[0]].copy(), X[idx[1]].copy(), X[idx[2]].copy()

X_alpha, X_beta, X_delta = get_leaders(X, fitness_vals)

# --- GWO loop ---
for t in range(MaxIter):
    a = 2 - 2*t/MaxIter
    for i in range(N):
        for j in range(dim):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2*a*r1 - a; C1 = 2*r2
            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2*a*r1 - a; C2 = 2*r2
            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2*a*r1 - a; C3 = 2*r2

            D_alpha = abs(C1*X_alpha[j] - X[i,j])
            D_beta  = abs(C2*X_beta[j]  - X[i,j])
            D_delta = abs(C3*X_delta[j] - X[i,j])

            X1 = X_alpha[j] - A1*D_alpha
            X2 = X_beta[j]  - A2*D_beta
            X3 = X_delta[j] - A3*D_delta

            X[i,j] = (X1 + X2 + X3)/3

    fitness_vals = np.array([fitness(xi) for xi in X])
    X_alpha, X_beta, X_delta = get_leaders(X, fitness_vals)

print("Best loss found:", fitness_vals[np.argmin(fitness_vals)])

# --- Evaluate on test set ---
def nn_predict(x, weights):
    y_pred = nn_forward(x, weights)
    return np.argmax(y_pred, axis=1)

y_pred_test = nn_predict(X_test, X_alpha)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred_test == y_test_labels)
print("Test accuracy:", accuracy)
