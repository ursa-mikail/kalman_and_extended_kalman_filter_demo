"""
ekf_stepbystep.py
-------------------
A DETERMINISTIC, VERBOSE trace of the exact same Extended Kalman Filter used
in ekf_radar_tracking.py -- with all randomness removed and every
intermediate matrix (including the Jacobian H, recomputed every step)
printed out.

Run this and compare its output line-by-line against README.md Section 8 --
they use the same fixed inputs, so every number should match exactly.

Run:
    python3 ekf_stepbystep.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=False)


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Model setup (identical to ekf_radar_tracking.py)
# ---------------------------------------------------------------------------
dt = 0.1
u = np.array([[1.2], [0.6]])
accel_noise_mag = 0.05
range_noise_mag = 2.0
bearing_noise_mag = np.deg2rad(3)

Ax = np.array([[1, dt], [0, 1]])
A = np.block([[Ax, np.zeros((2, 2))],
              [np.zeros((2, 2)), Ax]])
Bx = np.array([[dt**2 / 2], [dt]])
B = np.block([[Bx, np.zeros((2, 1))],
              [np.zeros((2, 1)), Bx]])

Ex1 = accel_noise_mag ** 2 * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
Ex = np.block([[Ex1, np.zeros((2, 2))],
               [np.zeros((2, 2)), Ex1]])
R = np.diag([range_noise_mag**2, bearing_noise_mag**2])


def h(state):
    x, y = state[0, 0], state[2, 0]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return np.array([[r], [theta]])


def jacobian_H(state):
    x, y = state[0, 0], state[2, 0]
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    return np.array([
        [x / r,    0, y / r,    0],
        [-y / r2,  0, x / r2,  0],
    ])


def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


hr("MODEL MATRICES (fixed for all time steps)")
print("A =\n", A)
print("B =\n", B)
print("Ex (process noise covariance) =\n", Ex)
print("R (measurement noise covariance) =\n", R)

# ---------------------------------------------------------------------------
# Initial belief: quail starts ~5m east of the Ninja, moving away
# ---------------------------------------------------------------------------
s_est = np.array([[5.0], [0.0], [0.0], [0.0]])
P = Ex.copy() * 10

# Hand-chosen "sensor readings" [range, bearing] for 2 steps -- no randomness.
fixed_measurements = [(6.0, 0.05), (7.2, 0.09)]

hr("INITIAL BELIEF (before any measurement)")
print("s_est (x, vx, y, vy) =\n", s_est.ravel())
print("P =\n", P)

for k, (r_meas, theta_meas) in enumerate(fixed_measurements, start=1):
    hr(f"STEP {k}  (incoming measurement: range={r_meas}, bearing={theta_meas} rad)")

    # ---- PREDICT (linear -- motion model has no nonlinearity here) ----
    s_pred = A @ s_est + B @ u
    P_pred = A @ P @ A.T + Ex
    print("-- Predict --")
    print(f"s_pred = A @ s_est + B@u =\n{s_pred.ravel()}")
    print(f"P_pred =\n{P_pred}")

    # ---- linearize: Jacobian H evaluated at s_pred ----
    H = jacobian_H(s_pred)
    z_pred = h(s_pred)
    print("-- Linearize (recomputed fresh, every step) --")
    print(f"h(s_pred) = [range_pred, bearing_pred] =\n{z_pred.ravel()}")
    print(f"H = Jacobian of h at s_pred =\n{H}")

    # ---- UPDATE ----
    z = np.array([[r_meas], [theta_meas]])
    innovation = z - z_pred
    innovation[1, 0] = angle_wrap(innovation[1, 0])
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    s_est = s_pred + K @ innovation
    P = (np.eye(4) - K @ H) @ P_pred

    print("-- Update --")
    print(f"innovation (range, bearing) = z - h(s_pred) =\n{innovation.ravel()}")
    print(f"S = H@P_pred@H.T + R =\n{S}")
    print(f"K = P_pred @ H.T @ inv(S) =\n{K}")
    print(f"s_est = s_pred + K@innovation =\n{s_est.ravel()}")
    print(f"P = (I - K@H) @ P_pred =\n{P}")

hr("DONE")
print("Compare every matrix above against README.md Section 8 -- they")
print("should match exactly (both use the same fixed inputs).")
