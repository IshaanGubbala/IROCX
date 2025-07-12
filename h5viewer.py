import h5py, numpy as np, matplotlib.pyplot as plt

with h5py.File("/Users/ishaangubbala/Documents/IROCX/results/results 7.12 - 0950.h5", "r") as f:
    t   = f["time"][:]                     # (121,)
    h2o = f["h2o2_concentration"][:]       # (121, 40, 25)

# plot mean H₂O₂ versus time
plt.plot(t, h2o.mean(axis=(1,2)))
plt.xlabel("Time (s)")
plt.ylabel("Mean H₂O₂ (µM)")
plt.show()
