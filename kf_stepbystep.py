"""
kf_stepbystep.py
-----------------
A DETERMINISTIC, VERBOSE trace of the exact same Kalman Filter used in
kf_quail_tracking.py -- with all randomness removed and every intermediate
matrix printed out.

Why this file exists: kf_quail_tracking.py uses real random noise, so the
numbers it produces can never be exactly reproduced by hand in a README.
This script instead uses fixed, hand-chosen measurements for the first
3 timesteps, so you can run it, read the printed matrices, and check them
against the worked-by-hand math in README.md Section 4 -- number for
number, with nothing hidden.

Run:
    python3 kf_stepbystep.py
"""

import numpy as np

# NOTE: suppress=False on purpose. Several of these matrices (Ex, P, K) hold
# genuinely tiny numbers (1e-6 to 1e-8) because dt=0.1 and accel_noise_mag is
# small. Rounding them to a fixed number of decimals would silently print
# them as 0.000000 and hide exactly the detail we want you to see -- e.g.
# that the Kalman gain K really is on the order of 1e-8, not exactly zero.
np.set_printoptions(precision=4, suppress=False)


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Model setup (identical to kf_quail_tracking.py)
# ---------------------------------------------------------------------------
dt = 0.1
u = 1.5
accel_noise_mag = 0.05
vision_noise_mag = 10.0

A = np.array([[1, dt],
              [0, 1]])
B = np.array([[dt**2 / 2],
              [dt]])
C = np.array([[1, 0]])

Ez = vision_noise_mag ** 2
Ex = accel_noise_mag ** 2 * np.array([[dt**4/4, dt**3/2],
                                       [dt**3/2, dt**2]])

hr("MODEL MATRICES (fixed for all time steps)")
print("A =\n", A)
print("B =\n", B)
print("C =\n", C)
print("Ex (process noise covariance) =\n", Ex)
print("Ez (measurement noise variance) =", Ez)

# ---------------------------------------------------------------------------
# Initial belief
# ---------------------------------------------------------------------------
x_est = np.array([[0.0], [0.0]])
P = Ex.copy()

# Hand-chosen "measurements" for 3 steps -- no randomness at all.
# These stand in for whatever the Ninja's eyes happened to report.
fixed_measurements = [12.3, 5.8, 9.4]

hr("INITIAL BELIEF (before any measurement)")
print("x_est =\n", x_est)
print("P =\n", P)

for k, y_k in enumerate(fixed_measurements, start=1):
    hr(f"STEP {k}  (incoming measurement y_{k} = {y_k})")

    # ---- PREDICT ----
    x_pred = A @ x_est + B * u
    P_pred = A @ P @ A.T + Ex
    print("-- Predict --")
    print(f"x_pred = A @ x_est + B*u =\n{x_pred}")
    print(f"P_pred = A @ P @ A.T + Ex =\n{P_pred}")

    # ---- UPDATE ----
    innovation = y_k - (C @ x_pred).item()
    S = (C @ P_pred @ C.T).item() + Ez
    K = (P_pred @ C.T) / S

    x_est = x_pred + K * innovation
    P = (np.eye(2) - K @ C) @ P_pred

    print("-- Update --")
    print(f"innovation = y_{k} - C@x_pred = {y_k} - {(C @ x_pred).item():.6f} = {innovation:.6f}")
    print(f"S = C@P_pred@C.T + Ez = {(C @ P_pred @ C.T).item():.6f} + {Ez} = {S:.6f}")
    print(f"K = P_pred @ C.T / S =\n{K}")
    print(f"x_est = x_pred + K*innovation =\n{x_est}")
    print(f"P = (I - K@C) @ P_pred =\n{P}")

hr("DONE")
print("Compare every matrix above against README.md Section 4 -- they")
print("should match exactly (both use the same fixed inputs).")
