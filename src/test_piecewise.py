import numpy as np
import matplotlib.pyplot as plt
from utils import plot_piecewise_linear, fit_piecewise_linear, piecewise_linear


if __name__ == "__main__":

    x = np.linspace(0, 10, 1000)

    hinge_x, hinge_y = 2, 3
    slope_1 = 0.5
    slope_2 = 2

    y = piecewise_linear(x, hinge_x, hinge_y, slope_1, slope_2)
    # add noise
    y += np.random.normal(0, 0.5, y.shape)

    # fit the model
    params, r_squared = fit_piecewise_linear(x, y)

    print(
        f"{hinge_x}/{params[0]} {hinge_y}/{params[1]} {slope_1}/{params[2]} {slope_2}/{params[3]}"
    )

    fig, ax = plt.subplots()

    ax.plot(x, y, "o", label="data")
    plot_piecewise_linear(0, 10, *params, ax, label=f"R^2 = {r_squared:.2f}")

    ax.grid()
    ax.legend()
    plt.show()
