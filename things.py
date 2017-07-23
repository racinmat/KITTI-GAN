import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# numlines = 128
# cols = matplotlib.cm.jet(np.arange(numlines))
#
# for i in np.linspace(0, numlines - 1, numlines):
#     col_idx = int(i)
#     plt.plot(np.arange(numlines), np.tile([i], numlines), linewidth=4, markersize=1, color=cols[col_idx, 0:3])
#
# plt.show()


data = np.random.rand(100, 1)
# the histogram of the data
nbins = 5
# n, bins, patches = plt.hist(rz, nbins, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = plt.hist(x=data, bins=nbins)
# n, bins, patches = plt.hist(rz)

plt.xlabel('yaw angle')
plt.ylabel('frequency')
plt.title('frequency of yaw angles of bounding box')
plt.axis([-1, 2, 0, 50])
# plt.grid(True)

plt.show()
# plt.savefig('angles.png')

