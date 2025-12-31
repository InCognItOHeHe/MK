import numpy as np
import pandas as pd

X = pd.read_csv("War9_X.csv", sep=";", decimal=",", header=None).to_numpy()
Xp = pd.read_csv("War9_Xprime.csv", sep=";", decimal=",", header=None).to_numpy()

print("X shape :", X.shape)
print("X' shape:", Xp.shape)

U, S, Vt = np.linalg.svd(X, full_matrices=False)
Sigma_inv = np.diag(1.0 / S)

A = Xp @ Vt.T @ Sigma_inv @ U.T

print("\nMacierz A (DMD):")
print(A)

eigvals = np.linalg.eigvals(A)
print("\nWartości własne A:")
print(eigvals)
