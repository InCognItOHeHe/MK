import numpy as np
import matplotlib.pyplot as plt

PARAM_NB_SAMPLES = 30
PARAM_SEQ_LEN = 20
PARAM_INIT_W = 0.5
PARAM_ETA_P = 1.2
PARAM_ETA_N = 0.5

print("Biblioteki zaimportowane, parametry ustawione.\n")


def generate_dataset(num_sequences, time_steps):
    allowed_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    X = np.random.choice(allowed_values, size=(num_sequences, time_steps))
    t = np.sum(X, axis=1)
    return X, t


print("--- (b) Generowanie danych ---")
X, t = generate_dataset(PARAM_NB_SAMPLES, PARAM_SEQ_LEN)
print(f"X.shape: {X.shape}")
print(f"t.shape: {t.shape}\n")


def update_state(xk, sk, W):
    return sk * W + xk


def forward_states(X, W):
    batch_size, time_steps = X.shape
    S = np.zeros((batch_size, time_steps + 1))

    for k in range(time_steps):
        xk = X[:, k]
        sk = S[:, k]
        S[:, k + 1] = update_state(xk, sk, W)

    return S


def loss(y, t):
    return 0.5 * np.mean((y - t) ** 2)


print("--- (c) Funkcje Forward zdefiniowane ---\n")


def output_gradient(y, t):
    return (y - t) / len(t)


def backward_gradient(X, S, t, W):
    batch_size, time_steps = X.shape
    y = S[:, -1]
    grad_y = output_gradient(y, t)
    total_grad_W = 0.0
    grad_next_s = grad_y
    grad_over_time = np.zeros((batch_size, time_steps))

    for k in range(time_steps - 1, -1, -1):
        s_prev = S[:, k]
        ds_dW = s_prev
        step_grad_W = np.sum(grad_next_s * ds_dW)
        total_grad_W += step_grad_W
        grad_over_time[:, k] = grad_next_s
        grad_next_s = grad_next_s * W

    return total_grad_W, grad_over_time


print("--- (d) Funkcje BPTT zdefiniowane ---\n")

print("--- Weryfikacja Wymiarów i Logiki (T=3) ---")
T_test = 3
X_mini = np.array([[0.2, 0.4, 0.6]])
t_mini = np.array([1.2])
W_mini = 0.5

S_mini = forward_states(X_mini, W_mini)

print(f"X_mini.shape: {X_mini.shape}")
print(f"S_mini.shape: {S_mini.shape} (oczekiwane: 1, 4)")
print(f"Stany obliczone kodem: {S_mini[0]}")
print(f"Oczekiwane ręcznie:    [0.   0.2  0.5  0.85]")

grad_W_mini, grad_ot_mini = backward_gradient(X_mini, S_mini, t_mini, W_mini)
print(f"grad_over_time.shape: {grad_ot_mini.shape} (oczekiwane: 1, 3)\n")

print("--- (e) Gradient Check ---")
epsilon = 1e-4
W_check = 0.8

S_check = forward_states(X, W_check)
grad_analytic, _ = backward_gradient(X, S_check, t, W_check)

loss1 = loss(forward_states(X, W_check - epsilon)[:, -1], t)
loss2 = loss(forward_states(X, W_check + epsilon)[:, -1], t)
grad_numeric = (loss2 - loss1) / (2 * epsilon)

print(f"Gradient Analityczny: {grad_analytic:.6f}")
print(f"Gradient Numeryczny:  {grad_numeric:.6f}")
if abs(grad_analytic - grad_numeric) < 1e-4:
    print(">> Weryfikacja gradientu: OK")
else:
    print(">> Weryfikacja gradientu: BŁĄD")
print("")

print("--- (f) Wizualizacja Powierzchni Błędu ---")
weights_range = np.linspace(0, 2.0, 50)
losses = []
grads = []

for w_val in weights_range:
    s_temp = forward_states(X, w_val)
    y_temp = s_temp[:, -1]
    l_val = loss(y_temp, t)
    g_val, _ = backward_gradient(X, s_temp, t, w_val)

    losses.append(l_val)
    grads.append(g_val)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(weights_range, losses, label="Loss")
plt.axvline(x=1.0, color="r", linestyle="--", label="Minimum (W=1)")
plt.title("Powierzchnia błędu (Loss vs W)")
plt.xlabel("Waga W")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(weights_range, grads, label="Gradient", color="orange")
plt.axhline(y=0, color="k", linestyle="-")
plt.axvline(x=1.0, color="r", linestyle="--")
plt.title("Gradient vs W")
plt.xlabel("Waga W")
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n--- (g) Optymalizacja RProp ---")

W_curr = PARAM_INIT_W
step_size = 0.1
prev_grad = 0.0

epochs = 50
history_loss = []
history_W = []

print(f"Start RProp. W_init: {W_curr}, Cel: 1.0")

for epoch in range(epochs):
    S_curr = forward_states(X, W_curr)
    curr_loss = loss(S_curr[:, -1], t)
    grad, _ = backward_gradient(X, S_curr, t, W_curr)

    history_loss.append(curr_loss)
    history_W.append(W_curr)

    if grad * prev_grad > 0:
        step_size = min(step_size * PARAM_ETA_P, 50.0)
        delta_w = -np.sign(grad) * step_size
        W_curr += delta_w
        prev_grad = grad

    elif grad * prev_grad < 0:
        step_size = max(step_size * PARAM_ETA_N, 1e-6)
        prev_grad = 0

    else:
        delta_w = -np.sign(grad) * step_size
        W_curr += delta_w
        prev_grad = grad

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d}: Loss = {curr_loss:.6f}, W = {W_curr:.4f}, Grad = {grad:.4f}"
        )

print(f"Końcowa waga W: {W_curr:.6f} (Oczekiwana: ~1.00)")

plt.figure(figsize=(8, 5))
plt.plot(history_W, history_loss, "o-", markersize=4)
plt.title("Trajektoria uczenia (Loss vs W)")
plt.xlabel("Waga W")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
