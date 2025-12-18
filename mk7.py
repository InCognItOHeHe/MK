import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return np.sin((x + 3 * y) ** 2)


def gradient(x, y):
    common_term = np.cos((x + 3 * y) ** 2)

    df_dx = common_term * 2 * (x + 3 * y)

    df_dy = common_term * 6 * (x + 3 * y)

    return df_dx, df_dy


learning_rate = 0.0001
max_iterations = 50000
tolerance = 1e-6

start_x = 2.0
start_y = 2.0

x_curr, y_curr = start_x, start_y
path_x, path_y, path_z = [x_curr], [y_curr], [f(x_curr, y_curr)]

for i in range(max_iterations):
    grad_x, grad_y = gradient(x_curr, y_curr)

    x_next = x_curr - learning_rate * grad_x
    y_next = y_curr - learning_rate * grad_y

    x_next = np.clip(x_next, 1, 3)
    y_next = np.clip(y_next, 1, 3)

    path_x.append(x_next)
    path_y.append(y_next)
    path_z.append(f(x_next, y_next))

    if np.sqrt((x_next - x_curr) ** 2 + (y_next - y_curr) ** 2) < tolerance:
        print(f"Zbieżność osiągnięta w iteracji {i}")
        break

    x_curr, y_curr = x_next, y_next

print(f"Znalezione minimum: f({x_curr:.4f}, {y_curr:.4f}) = {f(x_curr, y_curr):.4f}")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

x_range = np.linspace(1, 3, 100)
y_range = np.linspace(1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.6)
ax.plot(
    path_x,
    path_y,
    path_z,
    color="red",
    marker="o",
    markersize=3,
    label="Ścieżka Gradientu",
    zorder=10,
)
ax.scatter(
    path_x[-1],
    path_y[-1],
    path_z[-1],
    color="black",
    s=50,
    label="Znalezione Minimum",
    zorder=11,
)

ax.set_title("Wizualizacja Gradient Descent - Wariant 9")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")
ax.legend()

plt.show()
