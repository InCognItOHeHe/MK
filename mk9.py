import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- KONFIGURACJA DLA WARIANTU 9 ---
BIT_LENGTH = 12
NUM_SAMPLES = 10000
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01

# Ustawienie urządzenia (CPU jest wystarczające do tego zadania)
device = torch.device("cpu")


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


def generate_data(num_samples, bit_length):
    X = []
    Y = []
    max_val = 2**bit_length

    for _ in range(num_samples):
        a = np.random.randint(0, max_val)
        b = np.random.randint(0, max_val)
        res = (a - b) % max_val  # Różnica modulo

        a_bin = binary_encode(a, bit_length)
        b_bin = binary_encode(b, bit_length)
        res_bin = binary_encode(res, bit_length)

        # Wejście: [LSB...MSB], cechy: [bit_a, bit_b]
        x_entry = np.column_stack((a_bin, b_bin))

        X.append(x_entry)
        Y.append(res_bin[..., np.newaxis])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# Generowanie danych
print("Generowanie danych...")
X_train_np, y_train_np = generate_data(NUM_SAMPLES, BIT_LENGTH)
X_test_np, y_test_np = generate_data(1000, BIT_LENGTH)

# Konwersja na Tensory
X_train = torch.tensor(X_train_np).to(device)
y_train = torch.tensor(y_train_np).to(device)
X_test = torch.tensor(X_test_np).to(device)
y_test = torch.tensor(y_test_np).to(device)


# --- DEFINICJA MODELU RNN ---
class BinarySubtractRNN(nn.Module):
    def __init__(self):
        super(BinarySubtractRNN, self).__init__()
        # input_size=2 (dwa bity wejściowe), hidden_size=16 (stan ukryty - pamięć pożyczki)
        self.rnn = nn.RNN(input_size=2, hidden_size=16, batch_first=True)
        # Warstwa decyzyjna (0 lub 1)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # rnn_out shape: (batch, seq_len, hidden_size)
        rnn_out, _ = self.rnn(x)
        # Przetwarzamy wyjście RNN przez warstwę liniową dla każdego kroku
        out = self.fc(rnn_out)
        return self.sigmoid(out)


model = BinarySubtractRNN().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- TRENING ---
print("\nRozpoczynam trening (PyTorch)...")
dataset = torch.utils.data.TensorDataset(X_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoka {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

# --- WERYFIKACJA ---
print("\n--- TESTOWANIE ---")
model.eval()
with torch.no_grad():
    # Wybieramy 5 losowych próbek
    indices = np.random.choice(len(X_test), 5, replace=False)
    X_sample = X_test[indices]
    y_true_sample = y_test[indices]

    y_pred_prob = model(X_sample)
    y_pred = (y_pred_prob > 0.5).int()

    def decode_bits(bits_tensor):
        bits = bits_tensor.cpu().numpy().flatten()
        val = 0
        for i, bit in enumerate(bits):
            val += bit * (2**i)
        return int(val)

    for i in range(5):
        val_a = decode_bits(X_sample[i][:, 0])
        val_b = decode_bits(X_sample[i][:, 1])
        true_diff = decode_bits(y_true_sample[i])
        pred_diff = decode_bits(y_pred[i])

        status = "OK" if true_diff == pred_diff else "BŁĄD"
        print(
            f"A: {val_a:4d} | B: {val_b:4d} | Oczekiwana: {true_diff:4d} | Wynik sieci: {pred_diff:4d} | {status}"
        )
