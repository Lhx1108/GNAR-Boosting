import numpy as np

london_vts = np.genfromtxt("uk_traffic\\london_vts.csv",delimiter=",")
london_vts_daily = np.genfromtxt("uk_traffic\\london_vts_daily.csv",delimiter=",")
london_dist_mat = np.genfromtxt("uk_traffic\\london_dists.csv",delimiter=",")
london_locs = np.genfromtxt("uk_traffic\\london_locs.csv",delimiter=",")