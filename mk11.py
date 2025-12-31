import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Input
from tensorflow.keras.utils import to_categorical


text = """General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability, and economics"""

text = text.lower()

chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)

print(f"Liczba znaków w tekście: {n_chars}")
print(f"Liczba unikalnych znaków (słownik): {n_vocab}")

seq_length = 40
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i : i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print(f"Liczba wzorców treningowych: {n_patterns}")

X = np.reshape(dataX, (n_patterns, seq_length, 1))
# Normalizacja (0-1)
X = X / float(n_vocab)
# One-hot encoding wyjścia
y = to_categorical(dataY)

model = Sequential()

model.add(Input(shape=(X.shape[1], X.shape[2])))

model.add(LSTM(256))

model.add(Dense(y.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

print("Rozpoczynam trening (cel: loss < 0.1)...")
model.fit(X, y, epochs=500, batch_size=64, verbose=1)

print("\n--- Trening zakończony ---")


def generate_text(model, start_string, generation_length=300):
    pattern = [char_to_int[char] for char in start_string.lower()]
    print(f'\nZiarno startowe: "{start_string}"')
    print("Generowany tekst: ", end="")

    for i in range(generation_length):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)

        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]

        print(result, end="")

        pattern.append(index)
        pattern = pattern[1 : len(pattern)]
    print("\n")


start_seed = text[0:40]
generate_text(model, start_seed)
