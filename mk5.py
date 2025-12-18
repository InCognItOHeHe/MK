import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

filename = "allFaces.mat"
k_target = 90.0


H = 192
W = 168
# =============================================


def load_and_preprocess_data(filename, H, W):
    if not os.path.exists(filename):
        print(f"Błąd: Nie znaleziono pliku '{filename}'.")
        exit()

    print(f"Ładowanie danych z {filename}...")
    mat_data = scipy.io.loadmat(filename)

    data_keys = [key for key in mat_data.keys() if not key.startswith("__")]

    if not data_keys:
        print("Błąd: Nie znaleziono żadnych zmiennych z danymi w pliku MAT.")
        exit()

    var_name = data_keys[0]
    X_raw = mat_data[var_name]
    print(f"Znaleziono zmienną: '{var_name}' o kształcie {X_raw.shape}")

    n_pixels = H * W
    n_samples, n_features = X_raw.shape

    if n_features == n_pixels:
        X = X_raw
    elif n_samples == n_pixels:
        print("Wykryto odwróconą strukturę danych. Wykonuję transpozycję...")
        X = X_raw.T
    else:
        print(f"\nBŁĄD WYMIARÓW!")
        print(
            f"Oczekiwano, że jeden z wymiarów macierzy danych będzie równy H*W ({H}*{W}={n_pixels})."
        )
        print(f"Tymczasem dane mają kształt {X_raw.shape}.")
        print("Sprawdź poprawność ustawień H i W na początku skryptu.")
        exit()

    print(
        f"Dane gotowe do PCA. Liczba zdjęć: {X.shape[0]}, Liczba cech (pikseli): {X.shape[1]}"
    )
    return X


# --- GŁÓWNA CZĘŚĆ PROGRAMU ---

X = load_and_preprocess_data(filename, H, W)

print("\nObliczanie PCA (to może chwilę potrwać)...")
pca = PCA()
pca.fit(X)

# 3. Analiza wariancji i znalezienie 'r'
cumsum = np.cumsum(pca.explained_variance_ratio_)

threshold = k_target / 100.0
r = np.argmax(cumsum >= threshold) + 1

current_variance = cumsum[r - 1] * 100
print("\n" + "=" * 40)
print(f" WYNIKI DLA k={k_target}%")
print("=" * 40)
print(f"Wymagana liczba wartości własnych (r): {r}")
print(f"Dokładna zachowana informacja przy r={r}: {current_variance:.2f}%")
print(f"Całkowita dostępna liczba komponentów: {len(cumsum)}")


# 4. Rekonstrukcja przykładowego zdjęcia
sample_idx = 0
original_flat = X[sample_idx]

X_centered = original_flat - pca.mean_
weights = np.dot(X_centered, pca.components_[:r].T)
reconstructed_flat = pca.mean_ + np.dot(weights, pca.components_[:r])

original_img = original_flat.reshape((H, W))
reconstructed_img = reconstructed_flat.reshape((H, W))
mean_face_img = pca.mean_.reshape((H, W))

# 5. Wizualizacja
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title(f"Oryginał (próbka {sample_idx})")
plt.imshow(original_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Rekonstrukcja\n(użyto r={r} eigenfaces, k>={k_target}%)")
plt.imshow(reconstructed_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Pierwsza 'Eigenface' (Główny komponent)")
plt.imshow(pca.components_[0].reshape((H, W)), cmap="gray")
plt.axis("off")

plt.tight_layout()
print("\nWyświetlanie wyniku. Zamknij okno wykresu, aby zakończyć.")
plt.show()
