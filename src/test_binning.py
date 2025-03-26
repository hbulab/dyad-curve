import numpy as np
import matplotlib.pyplot as plt

from utils import compute_binned_values

np.random.seed(0)


x = np.random.rand(100)
y = x.copy()

small_x = x < 0.1
small_y = y < 0.1

# multiply small values by 100
x[small_x] *= 1000
y[small_y] *= 1000

bin_centers_x, pdf_edges_x, means_y, stds_y, errors_y, _, _, _, n_values_y = (
    compute_binned_values(x, y, 20)
)
bin_centers_y, pdf_edges_y, means_x, stds_x, errors_x, _, _, _, n_values_x = (
    compute_binned_values(y, x, 20)
)

fig, ax = plt.subplots(1, 2)
ax[0].errorbar(bin_centers_x, means_y, yerr=errors_y, fmt="o")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("y(x)")

ax[1].errorbar(bin_centers_y, means_x, yerr=errors_x, fmt="o")
ax[1].set_xlabel("y")
ax[1].set_ylabel("x")
ax[1].set_title("x(y)")

plt.show()
