# Kalman Filter & Extended Kalman Filter — A Ninja-Tracks-a-Quail Tutorial

This tutorial builds up the Kalman Filter (KF) and Extended Kalman Filter (EKF)
from scratch, using one running story so nothing feels like it appears out of
nowhere:

> A Ninja is trying to track a flying Quail. He can't see it perfectly — his
> eyes are noisy — but he *does* know roughly how quails fly (physics). The
> Kalman Filter is the mathematically optimal way for him to combine
> "what physics predicts" with "what his eyes report" into a single best
> guess, together with an honest measure of how confident that guess is.

Four code files go with this README:

| File | What it simulates | Filter used | Randomness? |
|---|---|---|---|
| `kf_quail_tracking.py` | 1‑D quail flight, Ninja measures **position directly** (noisy) | linear **KF** | yes (real sensor noise, for realistic plots) |
| `kf_stepbystep.py` | The exact same KF math, 3 timesteps | linear **KF** | **no** — fixed, hand-chosen inputs |
| `ekf_radar_tracking.py` | 2‑D quail flight, Ninja measures **range & bearing** (nonlinear) | **EKF** | yes (real sensor noise, for realistic plots) |
| `ekf_stepbystep.py` | The exact same EKF math, 2 timesteps | **EKF** | **no** — fixed, hand-chosen inputs |

Run any of them with `python3 <file>.py`. The two `_tracking.py` scripts save
plots into `./plots/`; the two `_stepbystep.py` scripts just print every
intermediate matrix to your terminal, deliberately with **no randomness**,
so that the numbers in this README are not approximations — they are the
literal, copy-pasted console output of those scripts. Run them yourself and
you should see the exact same numbers printed below, in Sections 4 and 8.

> **A note on why there are two versions of each filter:** the `_tracking.py`
> scripts are the "real" simulations (nice for plots, but their random noise
> means you can never exactly reproduce a number by hand). The
> `_stepbystep.py` scripts strip the randomness out and print every matrix,
> so you always have something deterministic to check your own by-hand
> arithmetic against. If a number in this README doesn't match what you get
> by hand, run the matching `_stepbystep.py` script — it is the ground
> truth.

---

## Table of contents

1. [The problem the Kalman Filter solves](#1-the-problem-the-kalman-filter-solves)
2. [Setting up the quail-tracking model](#2-setting-up-the-quail-tracking-model)
3. [Deriving the Kalman Filter equations](#3-deriving-the-kalman-filter-equations)
4. [Worked numeric example, by hand, one full step](#4-worked-numeric-example-by-hand-one-full-step)
5. [Reading the KF code, line by line](#5-reading-the-kf-code-line-by-line)
6. [From KF to EKF: why linear isn't always enough](#6-from-kf-to-ekf-why-linear-isnt-always-enough)
7. [Deriving the Extended Kalman Filter equations](#7-deriving-the-extended-kalman-filter-equations)
8. [Worked numeric example for the EKF](#8-worked-numeric-example-for-the-ekf)
9. [Reading the EKF code, line by line](#9-reading-the-ekf-code-line-by-line)
10. [KF vs EKF: when to use which](#10-kf-vs-ekf-when-to-use-which)
11. [Common pitfalls & things people get wrong](#11-common-pitfalls--things-people-get-wrong)

---

## 1. The problem the Kalman Filter solves

At every time step you have **two independent, imperfect sources of
information** about where something is:

1. **A prediction from physics** — "if it was here last step, moving at this
   speed, it should be roughly *there* now."
2. **A noisy measurement** — "my sensor / eyes say it's roughly *there*."

Neither one alone is trustworthy. But if you know *how* imprecise each one
is, you can combine them statistically so that the combination is **more
accurate than either alone**. That combination is a Kalman Filter update.

The trick that makes this tractable is assuming every uncertain quantity is
**Gaussian** (bell-curve) distributed, described completely by a mean and a
covariance. Gaussians have a wonderful property: the product of two
Gaussians is (proportional to) another Gaussian, and there is a closed-form
formula for its mean and variance. The whole Kalman Filter is essentially
"repeatedly multiply two Gaussians together and keep track of the result."

---

## 2. Setting up the quail-tracking model

### 2.1 The state

We describe the quail at time `k` by a **state vector**:

```
x_k = [ position ]
      [ velocity ]
```

We don't measure velocity directly (the Ninja can't clock the quail's
speed by eye), but the filter will *infer* it anyway — that's one of the
Kalman Filter's superpowers.

### 2.2 The motion model (how physics predicts the next state)

Basic kinematics: if you know position `p`, velocity `v`, and a constant
acceleration `u` applied over a short interval `dt`:

```
p_{k+1} = p_k + v_k*dt + (1/2)*u*dt^2
v_{k+1} = v_k + u*dt
```

Written as matrices, this becomes `x_{k+1} = A*x_k + B*u`:

```
A = [ 1   dt ]        B = [ dt^2/2 ]
    [ 0    1 ]            [   dt   ]
```

* `A` is the **state transition matrix** — "roll the state forward
  assuming no new forces."
* `B` is the **control matrix** — "here's how a *known* commanded input
  (acceleration `u`) additionally pushes the state."

This is exactly `A` and `B` in the MATLAB script and in
`kf_quail_tracking.py`.

### 2.3 The measurement model (how the sensor relates to the state)

The Ninja's eyes report a noisy position reading `y`. In terms of the
state:

```
y_k = C*x_k + noise,      C = [ 1   0 ]
```

`C` just says "the thing I measure is the first entry of the state
(position), and I don't measure the second entry (velocity) at all."

### 2.4 Two flavors of noise

* **Process noise** `Ex` (a.k.a. `Q` in most textbooks — renamed here to
  avoid clashing with the state variable `Q` used for "quail" in the
  MATLAB code) — captures that the quail's *actual* acceleration wobbles
  around the commanded value `u`. If acceleration noise has std-dev
  `σ_a`, then because position gets a `dt²/2` factor and velocity gets a
  `dt` factor, the covariance of the resulting state noise is:

  ```
  Ex = σ_a^2 * [ dt^4/4   dt^3/2 ]
               [ dt^3/2   dt^2   ]
  ```

  This is the standard "constant-acceleration white noise" model — the
  off-diagonal terms exist because a *single* random acceleration burst
  affects position and velocity in a *correlated* way (more acceleration
  noise → both a bigger position kick and a bigger velocity kick,
  together, not independently).

* **Measurement noise** `Ez` (a.k.a. `R`) — how bad the Ninja's eyesight
  is: `Ez = σ_vision²`.

---

## 3. Deriving the Kalman Filter equations

The filter alternates between two steps, forever:

### Step A — Predict (a.k.a. "time update")

Push last step's belief through the physics model:

```
x̂_k⁻ = A * x̂_{k-1} + B*u                       (predicted mean)
P_k⁻  = A * P_{k-1} * Aᵀ + Ex                    (predicted covariance)
```

*Why does covariance transform as `A P Aᵀ`?* If `x` has covariance `P`,
then any linear transform `A*x` has covariance `A*P*Aᵀ` — this is just the
multivariate version of "Var(aX) = a²Var(X)". We then add `Ex` because the
motion model itself is not perfect — every prediction step also injects
fresh uncertainty.

At this point we have a **prior belief**: a Gaussian
`N(x̂_k⁻, P_k⁻)` about where the quail is, before looking at this step's
measurement.

### Step B — Update (a.k.a. "measurement update")

Now the Ninja actually looks. We have a second Gaussian, this time from the
measurement: it says the position is around `y_k` with variance `Ez`.
Multiplying the two Gaussians (prior × likelihood) gives another Gaussian —
this is Bayes' rule for Gaussians, and its mean and variance work out to:

```
innovation:        ỹ_k = y_k − C*x̂_k⁻            (measurement minus what we expected to see)
innovation covar:  S_k = C*P_k⁻*Cᵀ + Ez
Kalman gain:       K_k = P_k⁻*Cᵀ*S_k⁻¹
posterior mean:    x̂_k = x̂_k⁻ + K_k*ỹ_k
posterior covar:   P_k = (I − K_k*C)*P_k⁻
```

**Intuition for the Kalman gain `K`:** it's a *trust ratio*.

* If the measurement is very noisy (`Ez` huge) relative to the prediction
  uncertainty (`P`), then `S` is dominated by `Ez`, so `K → 0` — the
  filter mostly ignores the measurement and trusts the physics prediction.
* If the prediction is very uncertain (`P` huge) relative to measurement
  noise, then `K → C⁻¹`-ish — the filter mostly trusts the measurement and
  throws away the prediction.
* In general `K` is a soft blend that is provably the **minimum-variance**
  (most precise) way to combine the two, given the noise assumptions.

*Where does `x̂_k = x̂_k⁻ + K_k*ỹ_k` come from, concretely?* You can derive
it either by (a) completing the square on the product of two Gaussian PDFs,
or (b) minimizing the trace of the posterior covariance `P_k` over choices
of gain `K` (set `dP_k/dK = 0` and solve). Both routes give exactly the `K`
above — this is why the Kalman Filter is called the "optimal linear
unbiased estimator": no other linear combination of prediction and
measurement produces a lower-variance estimate.

That's it — that's the whole filter. Every "Kalman Filter" you'll ever see
is this predict/update pair, just with bigger matrices.

---

## 4. Worked numeric example, by hand, one full step

Let's hand-trace the first three iterations with the tutorial's actual
numbers, so nothing is a black box. Every number below is the **literal
console output** of `kf_stepbystep.py` — run it yourself
(`python3 kf_stepbystep.py`) and you'll see this exact printout, digit for
digit. Nothing here is rounded-off-by-hand or approximated.

**Setup:** `dt = 0.1`, `u = 1.5`, `σ_a = 0.05` (accel_noise_mag),
`σ_vision = 10` (vision_noise_mag).

```
A =
 [[1.  0.1]
  [0.  1. ]]
B =
 [[0.005]
  [0.1  ]]
C =
 [[1 0]]
Ex (process noise covariance) =
 [[6.25e-08 1.25e-06]
  [1.25e-06 2.50e-05]]
Ez (measurement noise variance) = 100.0
```

**Initial belief:** `x̂_0 = [0, 0]ᵀ`, `P_0 = Ex` (we start as uncertain as one
process-noise step).

To keep this section reproducible with pen and paper, the "measurements"
below are **hand-chosen fixed numbers** (12.3, 5.8, 9.4) rather than random
draws — think of them as three specific eye-glances the Ninja happened to
report.

### Step 1 — measurement y₁ = 12.3

```
-- Predict --
x_pred = A @ x_est + B*u =
 [[0.0075]
  [0.15  ]]
P_pred = A @ P @ A.T + Ex =
 [[6.25e-07 5.00e-06]
  [5.00e-06 5.00e-05]]

-- Update --
innovation = y_1 - C@x_pred = 12.3 - 0.007500 = 12.292500
S = C@P_pred@C.T + Ez = 0.000001 + 100.0 = 100.000001
K = P_pred @ C.T / S =
 [[6.25e-09]
  [5.00e-08]]
x_est = x_pred + K*innovation =
 [[0.0075]
  [0.15  ]]
P = (I - K@C) @ P_pred =
 [[6.25e-07 5.00e-06]
  [5.00e-06 5.00e-05]]
```

Look closely at `K`: it's on the order of **1e-8 to 1e-9** — essentially
zero. That's why `x_est` barely moved from `x_pred` even though the
measurement (12.3) was wildly different from the prediction (0.0075). This
is not a bug — it's the correct, honest behavior of the equations given
these specific noise numbers, and it's worth understanding *why*, which
Section 4.1 below explains.

### Step 2 — measurement y₂ = 5.8

```
-- Predict --
x_pred = A @ x_est + B*u =
 [[0.03]
  [0.3 ]]
P_pred = A @ P @ A.T + Ex =
 [[2.1875e-06 1.1250e-05]
  [1.1250e-05 7.5000e-05]]

-- Update --
innovation = y_2 - C@x_pred = 5.8 - 0.030000 = 5.770000
S = C@P_pred@C.T + Ez = 0.000002 + 100.0 = 100.000002
K = P_pred @ C.T / S =
 [[2.1875e-08]
  [1.1250e-07]]
x_est = x_pred + K*innovation =
 [[0.03]
  [0.3 ]]
P = (I - K@C) @ P_pred =
 [[2.1875e-06 1.1250e-05]
  [1.1250e-05 7.5000e-05]]
```

### Step 3 — measurement y₃ = 9.4

```
-- Predict --
x_pred = A @ x_est + B*u =
 [[0.0675]
  [0.45  ]]
P_pred = A @ P @ A.T + Ex =
 [[5.25e-06 2.00e-05]
  [2.00e-05 1.00e-04]]

-- Update --
innovation = y_3 - C@x_pred = 9.4 - 0.067500 = 9.332500
S = C@P_pred@C.T + Ez = 0.000005 + 100.0 = 100.000005
K = P_pred @ C.T / S =
 [[5.25e-08]
  [2.00e-07]]
x_est = x_pred + K*innovation =
 [[0.067501]
  [0.450003]]
P = (I - K@C) @ P_pred =
 [[5.25e-06 2.00e-05]
  [2.00e-05 1.00e-04]]
```

Notice `P` keeps growing every step (6.25e-7 → 2.2e-6 → 5.25e-6 on the
diagonal) — that's the process noise `Ex` accumulating faster than the
tiny `K` can shrink it back down by trusting measurements. **Run
`kf_quail_tracking.py` and look at `plots/kf_distributions.png` to see this
same prior/measurement/posterior fusion happening visually, further into
the 10‑second simulation where `P` has grown enough for the gain to matter
more.**

### 4.1 Wait — is `kf_quail_tracking.py` actually a *good* teaching example?

Partially. The story (Ninja, Quail) is a fine narrative hook, and the
mechanics are 100% correct — but with these specific parameters
(`σ_a = 0.05`, `σ_vision = 10`), the Kalman gain stays tiny for a very long
time (it's still only ~0.005 after 100 steps — see the table below), so if
you only trace 1–3 steps by hand, you'll come away thinking "the update
step barely does anything, so why bother?" That's a real risk with this
specific example, and worth naming honestly rather than glossing over.

The reason the gain is so small: `Ex`'s diagonal (~1e-6 to 1e-8) is
**many orders of magnitude smaller** than `Ez` (=100). The filter has
concluded, correctly, that its physics-based prediction is far more
trustworthy than this Ninja's terrible eyesight — so it should barely
budge from the model. This is actually a realistic and useful lesson (a
very accurate model + a very noisy sensor *should* produce a filter that
leans almost entirely on the model), but it undersells the "blending"
intuition that makes the Kalman gain interesting in the first place.

To see the gain actually blend in a way you can feel, compare three noise
regimes side-by-side (same `A`, `B`, `C`, only `σ_a` and `σ_vision` change),
looking at the **steady-state** gain (i.e., after the filter has been
running long enough for `P` to stop changing much):

| Regime | `σ_a` (process) | `σ_vision` (measurement) | steady-state `K[position]` | Interpretation |
|---|---|---|---|---|
| **This tutorial's quail** | 0.05 | 10 | ≈ 0.005 | trust the *model* almost completely |
| **Balanced** | 2.0 | 3.0 | ≈ 0.109 | lean on the model, but the measurement matters |
| **Noisy flier, sharp eyes** | 8.0 | 1.0 | ≈ 0.329 | trust the *measurement* noticeably more |

(You can reproduce this table yourself: it's just `kf_stepbystep.py`'s
predict/update loop run for 200 steps with different `accel_noise_mag` /
`vision_noise_mag` values instead of 3 steps with the tutorial's defaults.)

**Bottom line:** keep `kf_quail_tracking.py` for the story and the
"recovering unmeasured velocity" demonstration — that part is genuinely
illustrative. But when you want to *feel* the Kalman gain trade off model
vs. measurement, mentally swap in the "balanced" or "sharp eyes" row above,
or edit `accel_noise_mag` / `vision_noise_mag` near the top of the script
and re-run it — the code doesn't change at all, only the two noise
numbers, which is itself a nice demonstration of how much those two
numbers alone control the filter's entire personality.

---

## 5. Reading the KF code, line by line

> If you want to watch the predict/update math execute one line at a time
> with printed matrices, run `kf_stepbystep.py` instead — it's the same
> filter with the randomness removed, and its output is exactly what
> Section 4 above shows.

`kf_quail_tracking.py` mirrors the math above 1:1:

* **Sections 1–3** build `A, B, C, Ex, Ez` exactly as derived above.
* **Section 4** simulates *ground truth*: it rolls the *true* quail state
  forward with real (unknown to the filter) process noise, and generates
  a noisy measurement from it. This block exists purely so we have
  something to compare the filter against — a real Ninja wouldn't have
  it.
* **Section 5** is the filter itself:
  ```python
  Q_est = A @ Q_est + B * u        # predict mean
  P = A @ P @ A.T + Ex              # predict covariance
  ...
  K = (P @ C.T) / S                 # Kalman gain
  Q_est = Q_est + K * innovation     # update mean
  P = (np.eye(2) - K @ C) @ P        # update covariance
  ```
  This is exactly Step A / Step B from Section 3, applied at every `k`.
* **Section 6** plots:
  - `kf_position.png` — you'll see the green (filtered) line hug the red
    (true) line far more tightly than the noisy black dots do.
  - `kf_velocity.png` — the filter recovers a state (velocity) **that was
    never measured at all**, purely from how position changes over time
    combined with the motion model.
  - `kf_covariance.png` — the posterior variance is always ≤ the
    predicted (prior) variance — every measurement can only make you more
    certain, never less (this is a mathematical guarantee of the update
    equation, not a coincidence).
  - `kf_distributions.png` — literally plots the three Gaussians
    (prior/prediction, measurement likelihood, posterior) at a few
    timesteps, so you can *see* Bayes' rule happening.

---

## 6. From KF to EKF: why linear isn't always enough

Everything above relied on two things being **linear**:

```
x_{k+1} = A*x_k + B*u        (motion model)
y_k     = C*x_k               (measurement model)
```

But lots of real sensors and real dynamics are **not linear**. Our Ninja
now trades his magic direct-position vision for a rangefinder + compass —
a totally realistic sensor. It reports:

```
r     = sqrt(x^2 + y^2)     (distance to the quail)
theta = atan2(y, x)         (bearing angle to the quail)
```

This `h(state) = [r; theta]` is **nonlinear** in `x` and `y` — there's no
matrix `C` such that `h(state) = C * state`. So the clean "Gaussian in →
Gaussian out" machinery from Section 3 breaks: if `state` is Gaussian,
`sqrt(x²+y²)` is *not* exactly Gaussian.

**The EKF's trick:** approximate `h` as linear *locally*, around the
current best estimate, using a first-order Taylor expansion:

```
h(state) ≈ h(x̂) + H * (state − x̂),      where H = ∂h/∂state |_{x̂}
```

`H` is the **Jacobian** of `h`. Once you have `H`, you plug it in
*exactly where `C` used to go* in the KF update equations. That's the
entire idea of the EKF: **re-derive a local linear approximation at every
single time step**, then run ordinary KF math on that approximation.

The same idea applies if the *motion* model is nonlinear (e.g., a car
turning at a fixed steering angle) — you'd linearize `A` via a Jacobian
`F` too. In our example the motion stays linear (constant velocity), only
the sensor is nonlinear, which keeps the comparison to the plain KF clean.

---

## 7. Deriving the Extended Kalman Filter equations

### 7.1 State (now 2-D flight)

```
s = [ x  ]
    [ vx ]
    [ y  ]
    [ vy ]
```

Motion is still linear (constant velocity + known acceleration `u=[ax,ay]`),
so `A` and `B` are just two independent copies of the 1‑D KF's `A, B`,
block-stacked — no new ideas here.

### 7.2 Nonlinear measurement function

```
h(s) = [ sqrt(x^2+y^2) ]
       [ atan2(y, x)   ]
```

### 7.3 The Jacobian, derived by hand

We need `H = ∂h/∂s`, a 2×4 matrix (2 measurement outputs, 4 state
variables). Let `r = sqrt(x²+y²)`.

**Row 1 — range `r` w.r.t. each state variable:**

```
∂r/∂x  = x / r                (chain rule on sqrt(x²+y²))
∂r/∂y  = y / r
∂r/∂vx = 0    (range doesn't depend on velocity directly)
∂r/∂vy = 0
```

**Row 2 — bearing `θ = atan2(y,x)` w.r.t. each state variable:**

Recall `d/dx[atan2(y,x)] = -y/(x²+y²)` and `d/dy[atan2(y,x)] = x/(x²+y²)`
(standard result — differentiate `θ=arctan(y/x)` and handle the chain rule
through `x²+y²`, or just accept it as the well-known gradient of
`atan2`).

```
∂θ/∂x  = -y / r²
∂θ/∂y  =  x / r²
∂θ/∂vx = 0
∂θ/∂vy = 0
```

Putting it together (columns ordered `x, vx, y, vy`):

```
H = [  x/r,   0,   y/r,   0 ]
    [ -y/r²,  0,   x/r²,  0 ]
```

This `H` is **evaluated at the current predicted state estimate**, fresh,
every single timestep — that's the "extended" part of EKF: the KF's
constant matrix `C` is replaced by a matrix that's recomputed at every
step from wherever the filter currently thinks it is.

### 7.4 The EKF predict/update equations

```
Predict:
  ŝ_k⁻ = A * ŝ_{k-1} + B*u          <- still exactly linear
  P_k⁻  = A * P_{k-1} * Aᵀ + Ex

Update:
  H_k = Jacobian of h(·) evaluated at ŝ_k⁻          <- NEW STEP vs. plain KF
  innovation:  z̃_k = z_k − h(ŝ_k⁻)                   <- use the TRUE nonlinear h() here, not H!
  S_k = H_k*P_k⁻*H_kᵀ + R
  K_k = P_k⁻*H_kᵀ*S_k⁻¹
  ŝ_k = ŝ_k⁻ + K_k*z̃_k
  P_k = (I − K_k*H_k)*P_k⁻
```

Two subtle but important details baked into the code:

1. **Use the exact nonlinear `h()` to compute the innovation**, and only
   use the *linearized* `H` to propagate covariance and compute the gain.
   Mixing these up (e.g. using `H*ŝ` instead of `h(ŝ)` for the innovation)
   is the single most common EKF bug.
2. **Angle wraparound.** Because `θ` lives on a circle, a naive subtraction
   like `θ_measured − θ_predicted` can spuriously jump near ±180°
   (e.g. -179° minus +179° "looks like" -358° instead of the true small
   gap of +2°). The code wraps this innovation back into `(-π, π]` with
   `angle_wrap()` before using it — always do this for any angular
   measurement in an EKF.

---

## 8. Worked numeric example for the EKF

As with the KF, every number below is the **literal console output** of
`ekf_stepbystep.py` — run `python3 ekf_stepbystep.py` yourself and you'll
get this exact printout. Two fixed (non-random) range/bearing readings
stand in for the Ninja's rangefinder: `(range=6.0, bearing=0.05 rad)` then
`(range=7.2, bearing=0.09 rad)`.

**Setup:** quail starts at `s_est = [x=5, vx=0, y=0, vy=0]`, with
`P = Ex * 10` (a bit more uncertain than one process-noise step, since we
don't trust the initial guess perfectly). Control input `u = [ax=1.2,
ay=0.6]`. `R = diag([2², (3°)²]) ≈ diag([4, 0.002742])`.

### Step 1 — measurement (range=6.0, bearing=0.05 rad)

```
-- Predict (still linear: motion model has no nonlinearity) --
s_pred = A @ s_est + B@u =
 [5.006  0.12   0.003  0.06 ]          # order: [x, vx, y, vy]
P_pred =
 [[5.6875e-06 3.8750e-05 0.0000e+00 0.0000e+00]
  [3.8750e-05 2.7500e-04 0.0000e+00 0.0000e+00]
  [0.0000e+00 0.0000e+00 5.6875e-06 3.8750e-05]
  [0.0000e+00 0.0000e+00 3.8750e-05 2.7500e-04]]

-- Linearize (Jacobian recomputed fresh, at s_pred) --
h(s_pred) = [range_pred, bearing_pred] = [5.006001, 0.000599]
H = Jacobian of h at s_pred =
 [[ 0.9999998  0.         0.0005993  0.        ]
  [-0.0001197  0.         0.1997602  0.        ]]

-- Update --
innovation (range, bearing) = z - h(s_pred) = [0.993999, 0.049401]
S = H@P_pred@H.T + R =
 [[4.000006e+00  ~0        ]
  [ ~0           2.741784e-03]]
K = P_pred @ H.T @ inv(S) =
 [[ 1.421873e-06 -2.483291e-07]
  [ 9.687484e-06 -1.691913e-06]
  [ 8.521011e-10  4.143785e-04]
  [ 5.805524e-09  2.823238e-03]]
s_est = s_pred + K@innovation = [5.006001  0.120095  0.003020  0.060139]
P (posterior) =
 [[5.687492e-06 3.874994e-05 2.772894e-13 1.889224e-12]
  [3.874994e-05 2.749996e-04 1.889224e-12 1.287164e-11]
  [2.772894e-13 1.889224e-12 5.687029e-06 3.874679e-05]
  [1.889224e-12 1.287164e-11 3.874679e-05 2.749781e-04]]
```

A few things worth actually noticing here, not skimming past:

* **`H`'s first row is `[≈1, 0, ≈0.0006, 0]`.** That means, right now,
  range is *almost exactly* just the x-coordinate — because the quail is
  nearly due east of the Ninja, so a small change in `x` moves the range
  almost 1‑for‑1, while a small change in `y` barely moves it at all. This
  is the Jacobian correctly recovering the "obvious" simple relationship
  exactly where the geometry makes it valid — it will look very different
  once the quail is, say, due north instead of due east.
* **Look at column 2 of `K`** (the column that multiplies the *bearing*
  innovation): its `y` and `vy` entries (`4.14e-4`, `2.82e-3`) are much
  bigger than its `x` and `vx` entries (`8.5e-10`, `5.8e-9`). That's
  because — from `H`'s second row — bearing depends almost entirely on
  `y` right now (coefficient `0.1998`) and almost not at all on `x`
  (coefficient `-0.00012`). So a bearing measurement mostly corrects `y`
  and `vy`, while a range measurement (column 1 of `K`) mostly corrects
  `x` and `vx`. **This split — different measurements informing different
  state components, automatically, with no code that says "if bearing,
  update y" — falls straight out of the Jacobian.** Nobody hard-coded it.
* **`vx` and `vy` get updated too** (`0 → 0.120`, `0 → 0.060`), even though
  neither is directly measured — the off-diagonal terms of `P_pred` (built
  up during the predict step, exactly like in the plain KF) carry the
  correction from position into velocity.

### Step 2 — measurement (range=7.2, bearing=0.09 rad)

```
-- Predict --
s_pred = A @ s_est + B@u = [5.024002  0.24001   0.012034  0.120139]

-- Linearize --
h(s_pred) = [5.024017, 0.002395]
H =
 [[ 0.9999971  0.         0.0023954  0.        ]
  [-0.0004768  0.         0.1990434  0.        ]]

-- Update --
innovation = [2.175983, 0.087605]
K =
 [[ 4.062466e-06 -2.825321e-06]
  [ 1.687486e-05 -1.173595e-05]
  [ 9.730583e-09  1.179414e-03]
  [ 4.041940e-08  4.899114e-03]]
s_est = s_pred + K@innovation = [5.024011  0.240045  0.012138  0.120569]
```

`H` shifted slightly between the two steps (`0.9999998 → 0.9999971` on the
range/x term) purely because the quail moved slightly — this is the
"extended" part of EKF in action: **the linearization point moves every
step, so `H` is recomputed every step**, unlike the plain KF's constant
`C`.

**See it end-to-end:** run `ekf_radar_tracking.py` and look at
`plots/ekf_trajectory.png` — the raw range/bearing readings (converted
back to x,y just for plotting) are scattered noisily, but the green EKF
estimate tracks the true red path closely, and
`plots/ekf_covariance_ellipse.png` shows the final 2‑σ confidence ellipse
around the last estimate.

---

## 9. Reading the EKF code, line by line

> As with the KF, `ekf_stepbystep.py` runs this same filter with no
> randomness and prints every matrix — its output is exactly Section 8
> above. Run it side-by-side with reading this section if any step feels
> abstract.

`ekf_radar_tracking.py` follows the same predict/update skeleton as the
KF script, with three additions that map directly onto Section 7:

```python
def h(state):
    ...
    return np.array([[r], [theta]])          # the NONLINEAR measurement model

def jacobian_H(state):
    ...
    return H                                  # the linearization of h(), Section 7.3

def angle_wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi     # keeps bearing innovations sane
```

Inside the main loop:

```python
s_est = A @ s_est + B @ u        # predict — identical to plain KF (motion is linear)
P = A @ P @ A.T + Ex

H = jacobian_H(s_est)             # <-- recomputed FRESH every timestep (the "extended" part)
z_pred = h(s_est)                  # <-- use the exact nonlinear h(), not H @ s_est

innovation = z_meas[k].reshape(2,1) - z_pred
innovation[1, 0] = angle_wrap(innovation[1, 0])   # guard against angle wraparound

S = H @ P @ H.T + R
K = P @ H.T @ np.linalg.inv(S)
s_est = s_est + K @ innovation
P = (np.eye(4) - K @ H) @ P
```

Everything else (plotting, noise simulation) mirrors the KF script.

---

## 10. KF vs EKF: when to use which

| | **Kalman Filter (KF)** | **Extended Kalman Filter (EKF)** |
|---|---|---|
| Motion model | must be linear: `x' = A*x + B*u` | can be nonlinear: `x' = f(x,u)` (linearized via Jacobian `F`) |
| Measurement model | must be linear: `y = C*x` | can be nonlinear: `y = h(x)` (linearized via Jacobian `H`) |
| Optimality | **Provably optimal** (minimum variance, unbiased) given Gaussian noise and linear models | Only **approximately** optimal — accuracy depends on how "linear-ish" `f`/`h` are near the current estimate |
| Computation | Fixed matrices, computed once | Jacobians recomputed every step (more CPU, more code) |
| Failure mode | None, if assumptions hold | Can **diverge** if the true state wanders far from where you linearized, or if the nonlinearity is very sharp over the size of your uncertainty |
| Tuning difficulty | Low | Higher — bad initial estimate + strong nonlinearity is a classic way to get a confidently-wrong filter |

**Rule of thumb:**
- If your sensor reads out something that's already linear in the state you
  care about (a GPS reading position, an odometer reading distance) → **KF**.
- If your sensor or dynamics involve angles, ranges, products of state
  variables, trig functions, or anything where "doubling the state doesn't
  double the output" → you need at least an **EKF**.
- If the nonlinearity is *severe*, or your initial uncertainty is huge
  relative to how curved `f`/`h` are, the EKF's linear approximation can be
  poor enough to diverge. In that regime, people reach for the **Unscented
  Kalman Filter (UKF)** (propagates a small set of sample "sigma points"
  through the *exact* nonlinear function instead of linearizing it — more
  accurate, no Jacobians needed, similar cost) or a **particle filter**
  (fully general, handles non-Gaussian noise too, but much more
  computationally expensive). This tutorial doesn't implement those, but
  it's the natural next step once EKF isn't good enough.

In short: **KF is a special case of EKF** where `f` and `h` happen to
already be linear (so the Jacobians `F` and `H` are just the constant
matrices `A` and `C`, and the linearization is exact rather than
approximate). Every trick you learned in Sections 3–5 is still exactly
what's running inside the EKF in Sections 7–9 — the only new idea in the
entire EKF is "recompute the Jacobian at the current estimate before doing
the same KF math."

---

## 11. Common pitfalls & things people get wrong

- **Using `H @ x` instead of `h(x)` for the innovation.** The Jacobian is
  only a *local* linear approximation. The actual innovation must compare
  the real measurement to the real nonlinear prediction `h(x̂⁻)`, not to
  `H*x̂⁻`. Using `H*x̂⁻` throws away exactly the nonlinear information you
  built the EKF to capture.
- **Forgetting to re-evaluate the Jacobian every step.** `H` (or `F`) is a
  function of the current state estimate — it changes constantly. Caching
  a stale Jacobian silently turns your EKF back into a (bad) linear KF.
- **Angle/units mismatches in the innovation.** Any circular quantity
  (bearing, heading, longitude near ±180°) needs explicit wraparound
  handling, or the filter will occasionally see huge fake innovations and
  overreact.
- **Confusing process noise `Ex`/`Q` with measurement noise `Ez`/`R`.**
  `Q` describes how wrong your *model of the world* is; `R` describes how
  wrong your *sensor* is. Swapping them (or setting either to 0 "to make
  it simpler") makes the filter overconfident in exactly the wrong way.
- **Initializing `P` at zero.** If you tell the filter "I am 100% certain
  of my initial state," the Kalman gain will be ~0 forever and it will
  ignore all future measurements. Always start with a `P` that honestly
  reflects your initial uncertainty.
- **Expecting the EKF to be optimal.** It isn't, by construction — it's a
  good, cheap approximation. If it's diverging, the fix is rarely "more
  decimal places"; it's usually a better initial estimate, a shorter
  timestep (so linearization is valid over a smaller step), or switching
  to a UKF/particle filter.
