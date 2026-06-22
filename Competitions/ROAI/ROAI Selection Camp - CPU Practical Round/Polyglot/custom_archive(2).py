"""
Baseline: assume rows already line up (identity). Meant as API/template only.
"""

from pathlib import Path

import numpy as np

FIXED_PATH = "./data/embeddings.npy"
TASKS = {
    1: "./data/subtask1.npy",
    2: "./data/subtask2.npy",
}
N = 4000
OUT_SUBMISSION = "./submission.csv"

M1 = np.load(FIXED_PATH)
f = open(OUT_SUBMISSION, "w")
f.write("subtaskID,datapointID,answer\n")

for subtask_id, path in TASKS.items():
    M2 = np.load(path)
    pred = np.arange(N, dtype=np.int64)
    for i in range(N):
        f.write(f"{subtask_id},{i},{pred[i]}\n")

f.close()
print(f"Wrote {OUT_SUBMISSION}")
