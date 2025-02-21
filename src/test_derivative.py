import numpy as np
import matplotlib.pyplot as plt


def my_derivative(x, t):

    dt = np.diff(t)
    dx = np.diff(x)
    dx_dt_forward = dx / dt

    dx_dt_central = np.zeros_like(x)
    dx_dt_central[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    dx_dt_central[0] = (x[1] - x[0]) / (t[1] - t[0])
    dx_dt_central[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])

    dx2_dt2 = np.zeros_like(x)
    dx2_dt2[1:-1] = 2 * (dx_dt_forward[1:] - dx_dt_forward[:-1]) / (dt[1:] + dt[:-1])
    dx2_dt2[0] = (dx_dt_central[1] - dx_dt_central[0]) / (t[1] - t[0])
    dx2_dt2[-1] = (dx_dt_central[-1] - dx_dt_central[-2]) / (t[-1] - t[-2])

    return dx_dt_central, dx2_dt2


if __name__ == "__main__":
    a = 0.5  # acceleration, m/s^2
    t = np.linspace(0, 10, 200)

    x = 0.5 * a * t**2

    dx_dt = np.gradient(x, t)
    d2x_dt2 = np.gradient(dx_dt, t)

    print(f"Using np.gradient:")
    print(f"dx/dt = {dx_dt}")
    print(f"d2x/dt2 = {d2x_dt2}")

    my_dx_dt, my_d2x_dt2 = my_derivative(x, t)

    print(f"Using my_derivative:")
    print(f"dx/dt = {dx_dt}")
    print(f"d2x/dt2 = {d2x_dt2}")

    fig, ax = plt.subplots(2, 1, figsize=(5, 5))

    ax[0].plot(t, dx_dt, label="dx/dt")
    ax[0].plot(t, my_dx_dt, label="my dx/dt")
    ax[0].set_xlabel("$t$ [s]")
    ax[0].set_ylabel("$dx/dt$ [m/s]")
    ax[0].legend()

    ax[1].plot(t, d2x_dt2, label="d2x/dt2")
    ax[1].plot(t, my_d2x_dt2, label="my d2x/dt2")
    ax[1].set_xlabel("$t$ [s]")
    ax[1].set_ylabel("$d2x/dt2$ [m/s^2]")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
