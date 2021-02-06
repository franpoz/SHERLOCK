import glob
import logging
import math
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ellc import ellc_f


def equal(a, b, tolerance=0.01):
    return np.abs(a - b) < tolerance

df = pd.read_csv("/home/martin/git_repositories/sherlockpipe/run_tests/experiment/a_tls_report.csv")
bls_df = pd.read_csv("/home/martin/git_repositories/sherlockpipe/run_tests/experiment/a_bls_report.csv")
min_period = df["period"].min()
max_period = df["period"].max()
min_rad = df["radius"].min()
max_rad = 3.99 #df["radius"].max()
period_grid = np.around(np.arange(min_period, max_period + 0.1, 0.5), 1)
radius_grid = np.around(np.arange(min_rad, max_rad + 0.1, 0.1), 1)
bresult = np.zeros((len(period_grid), len(radius_grid)))
for i in period_grid:
    ipos = int(round((i - min_period) * 2))
    for j in radius_grid:
        jpos = int(round((j - min_rad) * 10))
        sel_df = bls_df[equal(bls_df["period"], i)]
        sel_df = sel_df[equal(sel_df["radius"], j)]
        found_count = len(sel_df[sel_df["found"]])
        bresult[ipos][jpos] = found_count
result = np.zeros((len(period_grid), len(radius_grid)))
for i in period_grid:
    ipos = int(round((i - min_period) * 2))
    for j in radius_grid:
        jpos = int(round((j - min_rad) * 10))
        sel_df = df[equal(df["period"], i)]
        sel_df = sel_df[equal(sel_df["radius"], j)]
        found_count = len(sel_df[sel_df["found"]])
        result[ipos][jpos] = found_count
sdf = pd.read_csv("/home/martin/git_repositories/sherlockpipe/run_tests/experiment/a_sherlock_report.csv")
sresult = np.zeros((len(period_grid), len(radius_grid)))
for index, row in df.iterrows():
    entry = sdf[equal(sdf["period"], row["period"])]
    entry = entry[equal(entry["radius"], row["radius"])]
    entry = entry[equal(entry["epoch"], row["epoch"])]
    if (len(entry) > 0 and entry.iloc[0]["found"]) or row["found"]:
        ipos = int(round((row["period"] - min_period) * 2))
        jpos = int(round((row["radius"] - min_rad) * 10))
        sresult[ipos][jpos] = sresult[ipos][jpos] + 1
len_radius_grid = len(result[0]) - 8
result = result[:, :len_radius_grid]
sresult = sresult[:, :len_radius_grid]
bresult = bresult[:, :len_radius_grid]
diffresult = sresult - result
result = np.transpose(result)
sresult = np.transpose(sresult)
bresult = np.transpose(bresult)
fig, ax = plt.subplots()
im = ax.imshow(sresult)
ax.set_xticks(np.arange(len(period_grid)))
ax.set_yticks(np.arange(len_radius_grid))
ax.set_xticklabels(period_grid)
ax.set_yticklabels(radius_grid[:len_radius_grid])
ax.set_xlabel("Period")
ax.set_ylabel("Radius")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Found transits count", rotation=-90, va="bottom")
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.gca().invert_yaxis()
ax.set_title("SHERLOCK Period/radius recovery")
fig.tight_layout()
plt.savefig("a_sherlock_report.png")
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(sresult - result)
ax.set_xticks(np.arange(len(period_grid)))
ax.set_yticks(np.arange(len_radius_grid))
ax.set_xticklabels(period_grid)
ax.set_yticklabels(radius_grid[:len_radius_grid])
ax.set_xlabel("Period")
ax.set_ylabel("Radius")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Found transits difference", rotation=-90, va="bottom")
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.gca().invert_yaxis()
ax.set_title("Recovery SHERLOCK vs TLS (SNR>5)")
fig.tight_layout()
plt.savefig("a_difftls_report.png")
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(sresult - bresult)
ax.set_xticks(np.arange(len(period_grid)))
ax.set_yticks(np.arange(len_radius_grid))
ax.set_xticklabels(period_grid)
ax.set_yticklabels(radius_grid[:len_radius_grid])
ax.set_xlabel("Period")
ax.set_ylabel("Radius")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Found transits difference", rotation=-90, va="bottom")
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.gca().invert_yaxis()
ax.set_title("Recovery SHERLOCK vs BLS (SNR>7)")
fig.tight_layout()
plt.savefig("a_diffbls_report.png")
plt.show()

result_boolean = result == 0
sresult_boolean = sresult == 0
bresult_boolean = bresult == 0
result_limit = [next(key - 1 for key, value in enumerate(radius_array) if not value) for radius_array in np.transpose(result_boolean)]
sresult_limit = [next(key - 1 for key, value in enumerate(radius_array) if not value) for radius_array in np.transpose(sresult_boolean)]
bresult_limit = [next(key - 1 for key, value in enumerate(radius_array) if not value) for radius_array in np.transpose(bresult_boolean)]
plt.plot(period_grid, np.array(bresult_limit)[:19] / 10 + min_rad, 'r', label="BLS (SNR>7)")
plt.plot(period_grid, np.array(result_limit)[:19] / 10 + min_rad, 'b', label="TLS (SNR>5)")
plt.plot(period_grid, np.array(sresult_limit)[:19] / 10 + min_rad, 'g', label="SHERLOCK (SNR>7)")
plt.xticks(period_grid)
plt.yticks(radius_grid[:19])
plt.legend()
plt.xlabel("Period (days)")
plt.ylabel("Radius (R⊕)")
plt.title("Blind detection radius limit")
plt.savefig("a_blind_report.png")
plt.show()
plt.close()