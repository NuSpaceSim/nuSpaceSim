# The Clear BSD License
#
# Copyright (c) 2023 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Vectorized implementation of Dormand, Prince RKF45 ODE solver.

author: Alexander Reustle
date: 2023 May 25
"""

__all__ = ["dormand_prince_rk54"]

import numpy as np

# rk step coeffs
A = np.array(
    [
        [0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    ]
)
B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
E = np.array(
    [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
)

# Polynomial output for dense sampling along path.
P = np.array(
    [
        [
            1,
            -8048581381 / 2820520608,
            8663915743 / 2820520608,
            -12715105075 / 11282082432,
        ],
        [0, 0, 0, 0],
        [
            0,
            131558114200 / 32700410799,
            -68118460800 / 10900136933,
            87487479700 / 32700410799,
        ],
        [
            0,
            -1754552775 / 470086768,
            14199869525 / 1410260304,
            -10690763975 / 1880347072,
        ],
        [
            0,
            127303824393 / 49829197408,
            -318862633887 / 49829197408,
            701980252875 / 199316789632,
        ],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ]
)


def errnorm(K, h, scale):
    return np.abs((np.dot(E, K) * h / scale))


def initial_step(func, f0, t0, y0, rtol, atol, args):
    # Initial step
    scale = atol + np.abs(y0) * rtol
    d0 = np.abs(y0 / scale)
    d1 = np.abs(f0 / scale)
    h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 1e-2 * d0 / d1)
    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1, args)
    d2 = np.abs((f1 - f0) / scale) / h0
    h1 = np.where(
        (d1 <= 1e-15) | (d2 <= 1e-15),
        np.maximum(1e-6, h0 * 1e-3),
        (1e-2 / np.maximum(d1, d2)) ** (1 / 5),
    )
    h = np.minimum(100 * h0, h1)
    return h


def rk_rule(func, f, t, y, h, args, rtol, atol):
    """Runge Kutta Application."""

    # rk step
    # y_new, f_new
    K = np.empty((7, *y.shape), dtype=y.dtype)
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = func(t + c * h, y + dy, args)
    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = func(t + h, y_new, args)
    K[-1] = f_new

    # Error norm
    scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
    error_norm = errnorm(K, h, scale)

    return y_new, f_new, K, error_norm


def rk_step(func, f, t, y, h, tf, args, rtol, atol):
    max_step = np.inf
    min_step = 100 * np.abs(np.nextafter(1.0, np.inf) - 1.0)

    h = np.clip(h, min_step, max_step)
    K = np.empty((7, *y.shape), dtype=y.dtype)

    # incomplete, not accepted, unconverged.
    incomp = np.ones_like(y, dtype=np.bool_)
    # incomplete but about to be accepted.
    accept_mask = np.empty_like(y, dtype=np.bool_)
    iter = 0
    # breakpoint()
    while np.any(incomp):
        if np.any(h[incomp] < min_step):
            raise RuntimeError("Highly stiff region encountered, cannot proceed.")

        t_new = t[incomp] + h[incomp]
        t_new[t_new > tf[incomp]] = tf[incomp][t_new > tf[incomp]]
        _h = t_new - t[incomp]

        y_new, f_new, _K, err = rk_rule(
            func, f[incomp], t[incomp], y[incomp], _h, args[incomp], rtol, atol
        )

        K[..., incomp] = _K

        # Accepted steps are those where the error is less than 1.
        acc = err < 1.0
        # Rejected steps are those with large errors >= 1.
        accept_mask[incomp] = ~acc  # reuse the iacc buffer
        h[accept_mask] = _h[~acc] * np.maximum(0.2, 0.9 * err[~acc] ** -0.2)

        accept_mask[incomp] = acc  # Acceptance status along incomplete mask
        factor = np.zeros(np.count_nonzero(acc), dtype=h.dtype)
        # Steps with no error are limited to growing h by a factor of 10.
        zero_error = err[acc] == 0.0
        factor[zero_error] = 10.0
        factor[~zero_error] = np.minimum(10.0, 0.9 * err[acc][~zero_error] ** -0.2)
        # If a step was rejected previously, limit the h factor to at most 1.
        if iter > 0:
            factor[factor > 1.0] = 1.0

        h[accept_mask] = _h[acc] * factor
        t[accept_mask] = t_new[acc]
        y[accept_mask] = y_new[acc]
        f[accept_mask] = f_new[acc]
        incomp[accept_mask] = False  # Accepted. No longer incomplete
        accept_mask[accept_mask] = False
        iter += 1

    # interpolant coefficients
    Q = K.T.dot(P)

    return f, t, y, h, Q


def evaluate_t(t, t_old, y, h, Q):
    x = (t - t_old) / h
    # return y + h * x * (Q[..., 0] + x * (Q[..., 1] + x * (Q[..., 2] + x * Q[..., 3])))
    p = np.tile(x, (4, 1))
    p = np.cumprod(p, axis=0)
    return y + h * np.einsum("...i,i...->...", Q, p)


def dormand_prince_rk54(
    func,
    t_span,
    y0,
    args,
    t_eval=[],  # List of float or list of arrays
    rtol=1e-3,
    atol=1e-6,
    maxiter=100,
):
    t_span = np.array(t_span, copy=True)
    args = np.asarray(args)
    t0 = np.copy(t_span[0])
    tf = np.copy(t_span[1])
    t_eval = t_eval if t_eval else None
    t = np.copy(t0)
    y = np.array(y0, copy=True)
    f = func(t0, y0, args)
    h = initial_step(func, f, t0, y0, rtol, atol, args)

    ts = [np.zeros_like(i, dtype=t.dtype) for i in t_eval] if t_eval is not None else []
    ys = [np.zeros_like(i, dtype=y.dtype) for i in t_eval] if t_eval is not None else []

    # incomplete, not finished, still working
    icmp = np.ones_like(y, dtype=np.bool_)
    iter = 0
    while np.any(icmp) and iter < maxiter:

        t_old = np.copy(t[icmp])
        y_old = np.copy(y[icmp])

        # Do a step
        f[icmp], t[icmp], y[icmp], h[icmp], Q = rk_step(
            func, f[icmp], t[icmp], y[icmp], h[icmp], tf[icmp], args[icmp], rtol, atol
        )

        # breakpoint()

        if t_eval is None:
            ts.append(t[icmp])
            ys.append(y[icmp])

        else:
            dh = t[icmp] - t_old
            for e, out_t, out_y in zip(t_eval, ts, ys):
                m = (t_old < e[icmp]) & (e[icmp] <= t[icmp])
                if not np.any(m):
                    continue
                icmp_m = np.copy(icmp)
                icmp_m[icmp] = m
                out_t[icmp_m] = e[icmp_m]
                out_y[icmp_m] = evaluate_t(e[icmp_m], t_old[m], y_old[m], dh[m], Q[m])

        icmp = (t - tf) < 0
        iter += 1

    return ts, ys
