"""
EKF-Based GNSS Spoofing Detection
--------------------------------
This script demonstrates GNSS spoofing detection using EKF.

Includes:
- Trajectory simulation
- Spoofing attack injection
- FAILED vs IMPROVED EKF
- Multiple detection methods
- Multi-run evaluation (5 runs)

Author: Varun Wadhwa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SETTINGS
SPOOF_START = 50
N = 300
RUNS = 5
np.random.seed(42)

# -----------------------------
# TRAJECTORY
# -----------------------------
def generate_trajectory():
    gt = np.zeros((N,3))
    for i in range(1, N):
        gt[i] = gt[i-1] + np.random.normal([1, 0.5, 0], [0.2, 0.2, 0])
    return gt

# -----------------------------
# GPS + SPOOFING
# -----------------------------
def simulate_gps(gt):
    return gt + np.random.normal(0, 1.0, gt.shape)

def inject_spoof(gps):
    gps_sp = gps.copy()
    for i in range(SPOOF_START, len(gps)):
        gps_sp[i] += np.array([80, -60, 0])
    return gps_sp

# -----------------------------
# FAILED EKF (POSITION ONLY)
# -----------------------------
def basic_ekf(gps):
    x = np.zeros(3)
    P = np.eye(3)
    Q = np.eye(3)*0.05
    R = np.eye(3)

    res = []

    for z in gps:
        P += Q
        y = z - x
        S = P + R
        K = P @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(3)-K) @ P
        res.append(np.linalg.norm(y))

    return np.array(res)

# -----------------------------
# IMPROVED EKF
# -----------------------------
def ekf(gps):
    dt = 1
    x = np.zeros(6)
    P = np.eye(6)

    F = np.eye(6)
    F[0,3] = dt
    F[1,4] = dt

    H = np.zeros((3,6))
    H[0,0] = H[1,1] = H[2,2] = 1

    Q = np.eye(6)*0.1
    R = np.eye(3)

    traj, res = [], []

    for z in gps:
        x = F @ x
        P = F @ P @ F.T + Q

        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(6)-K@H) @ P

        traj.append(x[:3])
        res.append(np.linalg.norm(y))

    return np.array(traj), np.array(res)

# -----------------------------
# DETECTION METHODS
# -----------------------------
def fixed(res):
    return (res > 2).astype(int)

def adaptive(res):
    return (res > np.mean(res) + 2*np.std(res)).astype(int)

def jump(res):
    flags = np.zeros(len(res))
    for i in range(1, len(res)):
        if abs(res[i]-res[i-1]) > 3:
            flags[i] = 1
    return flags

# -----------------------------
# METRICS
# -----------------------------
def metrics(pred):
    labels = np.zeros(len(pred))
    labels[SPOOF_START:] = 1

    TP = np.sum((pred==1)&(labels==1))
    FP = np.sum((pred==1)&(labels==0))
    FN = np.sum((pred==0)&(labels==1))

    p = TP/(TP+FP+1e-6)
    r = TP/(TP+FN+1e-6)
    f1 = 2*p*r/(p+r+1e-6)

    return p, r, f1

# -----------------------------
# MULTI RUN
# -----------------------------
def run_experiment():
    results = {"Fixed": [], "Adaptive": [], "Jump": []}

    for _ in range(RUNS):
        gt = generate_trajectory()
        gps = simulate_gps(gt)
        gps_sp = inject_spoof(gps)

        traj, res = ekf(gps_sp)

        results["Fixed"].append(metrics(fixed(res)))
        results["Adaptive"].append(metrics(adaptive(res)))
        results["Jump"].append(metrics(jump(res)))

    return results

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    # FAILED RUN
    gt = generate_trajectory()
    gps = simulate_gps(gt)
    gps_sp = inject_spoof(gps)

    res_fail = basic_ekf(gps_sp)
    print("FAILED RUN:", metrics(fixed(res_fail)))

    # IMPROVED RUN
    results = run_experiment()

    avg = {k: np.mean(v, axis=0) for k,v in results.items()}

    print("\nIMPROVED (AVERAGED):")
    for k in avg:
        print(k, avg[k])

    # SAVE CSV
    df = pd.DataFrame({
        "Method": list(avg.keys()),
        "Precision": [avg[k][0] for k in avg],
        "Recall": [avg[k][1] for k in avg],
        "F1": [avg[k][2] for k in avg]
    })

    df.to_csv("metrics.csv", index=False)

    # FINAL VISUALS
    traj, res = ekf(gps_sp)

    plt.plot(gt[:,0], gt[:,1], label="GT")
    plt.plot(gps_sp[:,0], gps_sp[:,1], label="Spoofed")
    plt.plot(traj[:,0], traj[:,1], label="EKF")
    plt.legend()
    plt.title("Trajectory")
    plt.savefig("trajectory.png")

    plt.figure()
    plt.plot(res)
    plt.axvline(SPOOF_START)
    plt.title("Residual")
    plt.savefig("residual.png")

    plt.figure()
    plt.bar(df["Method"], df["F1"])
    plt.title("F1 Comparison")
    plt.savefig("f1_chart.png")