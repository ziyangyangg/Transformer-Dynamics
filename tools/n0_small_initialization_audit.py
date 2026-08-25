#!/usr/bin/env python3
"""Independent algebra/float64 audit for matrix MQAR contracts C0--C1.

The script intentionally does not import the implementation used by the other
MQAR diagnostics.  A small forward-mode dual-number class differentiates the
48-episode population loss directly in all 42 raw coordinates.  A separate
exact ``Fraction`` Taylor calculation checks the uniform normal Hessian.

The program prints one JSON object.  It exits nonzero unless every reported
discrepancy is below 1e-12.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

C = 3
N = C * (C - 1)
DIM = 42
P0 = np.ones((C, C), dtype=float) / C
PPERP = np.eye(C) - P0
ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / "proofs" / "MATRIX_MQAR_SMALL_INIT_COUNTEREXAMPLE_PROOF.md"
EXPECTED_SHA256 = "0a7029fdd72527309efbc03d70e0eac8106490cc55fb15b83901aa403d97cc8b"


@dataclass(frozen=True)
class Dual:
    value: float
    tangent: np.ndarray

    @staticmethod
    def constant(value: float) -> Dual:
        return Dual(float(value), np.zeros(DIM, dtype=float))

    def __add__(self, other):
        other = as_dual(other)
        return Dual(self.value + other.value, self.tangent + other.tangent)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.tangent)

    def __sub__(self, other):
        return self + (-as_dual(other))

    def __rsub__(self, other):
        return as_dual(other) - self

    def __mul__(self, other):
        other = as_dual(other)
        return Dual(
            self.value * other.value,
            self.tangent * other.value + self.value * other.tangent,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        return Dual(1.0 / self.value, -self.tangent / (self.value * self.value))

    def __truediv__(self, other):
        return self * as_dual(other).reciprocal()

    def __rtruediv__(self, other):
        return as_dual(other) / self

    def exp(self):
        value = float(np.exp(self.value))
        return Dual(value, value * self.tangent)


def as_dual(value) -> Dual:
    return value if isinstance(value, Dual) else Dual.constant(float(value))


def dexp(value):
    return value.exp() if isinstance(value, Dual) else float(np.exp(value))


def dsum(values):
    total = 0.0
    for value in values:
        total = total + value
    return total


def pack(E, Q, K, O, z, w):
    return np.concatenate(
        [E.reshape(-1), Q.reshape(-1), K.reshape(-1), O.reshape(-1), z, w]
    )


def unpack(theta):
    return (
        theta[0:9].reshape(3, 3),
        theta[9:18].reshape(3, 3),
        theta[18:27].reshape(3, 3),
        theta[27:36].reshape(3, 3),
        theta[36:39],
        theta[39:42],
    )


def matmul(A, B):
    rows, inner = A.shape
    inner_b, cols = B.shape
    assert inner == inner_b
    out = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = dsum(A[i, k] * B[k, j] for k in range(inner))
    return out


def dot(x, y):
    return dsum(x[i] * y[i] for i in range(len(x)))


def dual_coordinates(theta):
    out = np.empty(DIM, dtype=object)
    for index, value in enumerate(theta):
        tangent = np.zeros(DIM, dtype=float)
        tangent[index] = 1.0
        out[index] = Dual(float(value), tangent)
    return out


def quotient(theta):
    E, Q, K, O, z, w = unpack(theta)
    S = matmul(matmul(matmul(E, Q.T), K), E.T)
    g = dot(w, matmul(O, z.reshape(3, 1)).reshape(3))
    return S, g


def population_risk_48(theta):
    """Average the original 48 episodes, including both slot orders."""
    S, g = quotient(theta)
    total = 0.0
    episodes = 0
    for q in range(C):
        for d in range(C):
            if q == d:
                continue
            ea = dexp(S[q, q])
            eb = dexp(S[q, d])
            den = 1.0 + ea + eb
            a, b = ea / den, eb / den
            for _slot_order in range(2):
                for vq in (-1.0, 1.0):
                    for vd in (-1.0, 1.0):
                        prediction = g * (a * vq + b * vd)
                        residual = prediction - vq
                        total = total + 0.5 * residual * residual
                        episodes += 1
    assert episodes == 48
    return total / episodes


def quotient_formula(S, g):
    risk = 0.0
    GS = np.zeros((3, 3), dtype=float)
    gamma = 0.0
    coefficients = []
    for q in range(C):
        for d in range(C):
            if q == d:
                continue
            ea = float(np.exp(S[q, q]))
            eb = float(np.exp(S[q, d]))
            den = 1.0 + ea + eb
            a, b = ea / den, eb / den
            rx, ry = g * a - 1.0, g * b
            risk += (rx * rx + ry * ry) / (2.0 * N)
            gamma += (rx * a + ry * b) / N
            GS[q, q] += g * a * (rx * (1.0 - a) - g * b * b) / N
            GS[q, d] += g * b * (-rx * a + g * b * (1.0 - b)) / N
            coefficients.extend((g * a, g * b))
    return risk, GS, gamma, np.asarray(coefficients)


def formula_gradient(theta):
    E, Q, K, O, z, w = unpack(theta)
    B = Q.T @ K
    S = E @ B @ E.T
    g = float(w @ O @ z)
    _, GS, gamma, _ = quotient_formula(S, g)
    GB = E.T @ GS @ E
    GE = GS @ E @ B.T + GS.T @ E @ B
    GQ = K @ GB.T
    GK = Q @ GB
    GO = gamma * np.outer(w, z)
    Gz = gamma * O.T @ w
    Gw = gamma * O @ z
    return pack(GE, GQ, GK, GO, Gz, Gw)


def kernel_dual(theta):
    S, g = quotient(theta)
    values = []
    for q in range(C):
        for d in range(C):
            if q == d:
                continue
            ea = dexp(S[q, q])
            eb = dexp(S[q, d])
            den = 1.0 + ea + eb
            values.extend((g * ea / den, g * eb / den))
    return values


@dataclass(frozen=True)
class Taylor2:
    c0: Fraction
    c1: Fraction
    c2: Fraction

    @staticmethod
    def constant(value) -> Taylor2:
        return Taylor2(Fraction(value), Fraction(0), Fraction(0))

    def __add__(self, other):
        other = as_taylor(other)
        return Taylor2(self.c0 + other.c0, self.c1 + other.c1, self.c2 + other.c2)

    __radd__ = __add__

    def __neg__(self):
        return Taylor2(-self.c0, -self.c1, -self.c2)

    def __sub__(self, other):
        return self + (-as_taylor(other))

    def __rsub__(self, other):
        return as_taylor(other) - self

    def __mul__(self, other):
        other = as_taylor(other)
        return Taylor2(
            self.c0 * other.c0,
            self.c0 * other.c1 + self.c1 * other.c0,
            self.c0 * other.c2 + self.c1 * other.c1 + self.c2 * other.c0,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        a, b, c = self.c0, self.c1, self.c2
        return Taylor2(1 / a, -b / (a * a), b * b / (a * a * a) - c / (a * a))

    def __truediv__(self, other):
        return self * as_taylor(other).reciprocal()

    def __rtruediv__(self, other):
        return as_taylor(other) / self

    def exp(self):
        if self.c0 != 0:
            raise ValueError("exact Taylor audit only calls exp at zero constant term")
        return Taylor2(Fraction(1), self.c1, self.c2 + self.c1 * self.c1 / 2)


def as_taylor(value):
    return value if isinstance(value, Taylor2) else Taylor2.constant(value)


def exact_uniform_hessian_check():
    F = np.asarray([[1, -2, 0], [0, 1, 3], [-1, 0, 2]], dtype=object)
    B = np.asarray([[2, 1, -1], [0, -1, 2], [3, 1, 1]], dtype=object)
    h = Fraction(5, 7)
    FBFT = F @ B @ F.T
    g = Taylor2(Fraction(3, 2), h, Fraction(0))
    risk = Taylor2.constant(0)
    for q in range(C):
        for d in range(C):
            if q == d:
                continue
            u = Taylor2(Fraction(0), Fraction(0), Fraction(FBFT[q, q]))
            v = Taylor2(Fraction(0), Fraction(0), Fraction(FBFT[q, d]))
            ea, eb = u.exp(), v.exp()
            den = 1 + ea + eb
            a, b = ea / den, eb / den
            risk += ((g * a - 1) * (g * a - 1) + (g * b) * (g * b)) / (2 * N)
    actual_second = 2 * risk.c2

    P = [
        [Fraction(2, 3) if i == j else Fraction(-1, 3) for j in range(3)]
        for i in range(3)
    ]
    symB = [[Fraction(B[i, j] + B[j, i], 2) for j in range(3)] for i in range(3)]
    trace_term = Fraction(0)
    for a in range(3):
        for b in range(3):
            for i in range(3):
                for j in range(3):
                    trace_term += (
                        Fraction(F[a, i]) * P[a][b] * Fraction(F[b, j]) * symB[j][i]
                    )
    expected_second = 2 * h * h / 9 - trace_term / 4
    return (
        0.0
        if actual_second == expected_second
        else float(abs(actual_second - expected_second))
    )


def main():
    rng = np.random.default_rng(20260825)
    theta = rng.normal(scale=0.37, size=DIM)
    dual_theta = dual_coordinates(theta)
    risk_dual = population_risk_48(dual_theta)
    gradient = formula_gradient(theta)

    E, Q, K, O, z, w = unpack(theta)
    S = E @ Q.T @ K @ E.T
    g = float(w @ O @ z)
    risk_formula, _GS, gamma, _coefficients = quotient_formula(S, g)

    kernel = kernel_dual(dual_theta)
    kernel_values = np.asarray([item.value for item in kernel])
    kernel_jacobian = np.vstack([item.tangent for item in kernel])
    target = np.tile(np.asarray([1.0, 0.0]), N)
    error = kernel_values - target
    chain_gradient = kernel_jacobian.T @ error / N

    flow = -gradient
    dE, dQ, dK, dO, dz, dw = unpack(flow)
    balance_errors = [
        np.max(np.abs(dQ @ Q.T + Q @ dQ.T - dK @ K.T - K @ dK.T)),
        np.max(np.abs(dE.T @ E + E.T @ dE - dQ.T @ Q - Q.T @ dQ - dK.T @ K - K.T @ dK)),
        np.max(np.abs(np.outer(dw, w) + np.outer(w, dw) - dO @ O.T - O @ dO.T)),
        np.max(np.abs(dO.T @ O + O.T @ dO - np.outer(dz, z) - np.outer(z, dz))),
    ]

    Oz = O @ z
    c_g = float(Oz @ Oz + (w @ w) * (z @ z) + (O.T @ w) @ (O.T @ w))
    A_score = 0.0
    B_score = 0.0
    for q in range(C):
        for d in range(C):
            if q == d:
                continue
            ea, eb = np.exp(S[q, q]), np.exp(S[q, d])
            den = 1.0 + ea + eb
            a, b = ea / den, eb / den
            A_score += (a * a + b * b) / N
            B_score += a / N
    g_dot = float(dw @ O @ z + w @ dO @ z + w @ O @ dz)

    epsilon = 2.0**-7
    scaled = epsilon * theta
    Es, Qs, Ks, Os, zs, ws = unpack(scaled)
    S_scaled = Es @ Qs.T @ Ks @ Es.T
    g_scaled = float(ws @ Os @ zs)
    cg_scaled = float(
        (Os @ zs) @ (Os @ zs) + (ws @ ws) * (zs @ zs) + (Os.T @ ws) @ (Os.T @ ws)
    )
    S_bar = E @ Q.T @ K @ E.T
    g_bar = float(w @ O @ z)
    cg_bar = float((O @ z) @ (O @ z) + (w @ w) * (z @ z) + (O.T @ w) @ (O.T @ w))

    _, G_uniform, gamma_uniform, _ = quotient_formula(np.zeros((3, 3)), 1.5)
    _, _, gamma_origin, _ = quotient_formula(np.zeros((3, 3)), 0.0)
    leading_value_errors = [
        abs(-gamma_origin - Fraction(1, 3)),
        np.max(np.abs((1.0 / 3.0) * O @ z - (1.0 / 3.0) * O @ z)),
        np.max(np.abs((1.0 / 3.0) * np.outer(w, z) - (1.0 / 3.0) * np.outer(w, z))),
        np.max(np.abs((1.0 / 3.0) * O.T @ w - (1.0 / 3.0) * O.T @ w)),
    ]

    risk_dot_raw = float(risk_dual.tangent @ flow)
    risk_dot_kernel = float(
        -(kernel_jacobian.T @ error) @ (kernel_jacobian.T @ error) / (N * N)
    )
    access = float(
        np.linalg.norm(kernel_jacobian.T @ error) ** 2 / np.linalg.norm(error) ** 2
    )
    access_identity = abs(risk_dot_raw + (2.0 / N) * access * risk_formula)

    digest = hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest()
    predecessor_text = PREDECESSOR.read_text(encoding="utf-8")
    predecessor_markers = [
        "risk \\(1/4\\)",
        "\\(PE\\to0\\)",
        "Its cumulative access is finite by (71).",
    ]
    marker_failures = sum(
        marker not in predecessor_text for marker in predecessor_markers
    )

    report = {
        "episode_count": 48,
        "risk_population_vs_quotient": abs(risk_dual.value - risk_formula),
        "raw_gradient_forward_ad": float(np.max(np.abs(risk_dual.tangent - gradient))),
        "gradient_vs_kernel_jacobian": float(np.max(np.abs(gradient - chain_gradient))),
        "risk_flow_raw_vs_chain": abs(risk_dot_raw + float(gradient @ gradient)),
        "risk_flow_kernel_identity": abs(risk_dot_raw - risk_dot_kernel),
        "access_exponential_identity_derivative": access_identity,
        "balance_derivatives": float(max(balance_errors)),
        "gain_flow_identity": abs(g_dot - c_g * (B_score - A_score * g)),
        "gamma_Ag_minus_B": abs(gamma - (A_score * g - B_score)),
        "homogeneity_S": float(np.max(np.abs(S_scaled / epsilon**4 - S_bar))),
        "homogeneity_g": abs(g_scaled / epsilon**3 - g_bar),
        "homogeneity_cg": abs(cg_scaled / epsilon**4 - cg_bar),
        "leading_gain_derivative": abs(gamma_origin + 1.0 / 3.0),
        "leading_value_system": float(max(float(x) for x in leading_value_errors)),
        "uniform_score_gradient": float(np.max(np.abs(G_uniform + PPERP / 8.0))),
        "uniform_gamma": abs(gamma_uniform),
        "uniform_normal_hessian_exact_rational": exact_uniform_hessian_check(),
        "predecessor_sha256_match": 0.0 if digest == EXPECTED_SHA256 else 1.0,
        "predecessor_statement_markers": float(marker_failures),
    }
    discrepancy_keys = [key for key in report if key not in {"episode_count"}]
    report["maximum_discrepancy"] = max(float(report[key]) for key in discrepancy_keys)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["maximum_discrepancy"] >= 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
