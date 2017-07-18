import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure()


numlines = 128
cols = matplotlib.cm.jet(np.arange(numlines))

for i in np.linspace(0, numlines - 1, numlines):
    col_idx = int(i)
    plt.plot(np.arange(numlines), np.tile([i], numlines), linewidth=4, markersize=1, color=cols[col_idx, 0:3])

plt.show()
