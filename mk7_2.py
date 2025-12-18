import numpy as np


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


nn_architecture = [
    {"input_dim": 2, "output_dim": 2, "activation": "elu"},
    {"input_dim": 2, "output_dim": 1, "activation": "tanh"},
]


def init_layers(nn_architecture, seed=42):
    np.random.seed(seed)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values["W" + str(layer_idx)] = (
            np.random.randn(layer_output_size, layer_input_size) * 0.1
        )
        params_values["b" + str(layer_idx)] = (
            np.random.randn(layer_output_size, 1) * 0.1
        )

    return params_values


def forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]

        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activ_function == "elu":
            A_curr = elu(Z_curr)
        elif activ_function == "tanh":
            A_curr = tanh(Z_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory


def backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = 2 * (Y_hat - Y)  # / m

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]

        if activ_function == "elu":
            dZ_curr = dA_curr * elu_derivative(Z_curr)
        elif activ_function == "tanh":
            dZ_curr = dA_curr * tanh_derivative(Z_curr)

        dW_curr = np.dot(dZ_curr, A_prev.T)
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True)

        dA_prev = np.dot(W_curr.T, dZ_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


X = np.array([[1.5], [2.0]])

Y = np.array([[0.5]])

params = init_layers(nn_architecture)

Y_hat, cache = forward_propagation(X, params, nn_architecture)
print(f"Wynik sieci (Y_hat): {Y_hat[0][0]:.4f}")

grads = backward_propagation(Y_hat, Y, cache, params, nn_architecture)

print("\n--- Obliczone Gradienty ---")
print(f"Gradient dW2 (Warstwa wyjściowa - Tanh):\n{grads['dW2']}")
print(f"Gradient db2:\n{grads['db2']}")
print("-" * 20)
print(f"Gradient dW1 (Warstwa ukryta - ELU):\n{grads['dW1']}")
print(f"Gradient db1:\n{grads['db1']}")
