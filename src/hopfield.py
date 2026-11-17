"""Classical Hopfield network with Hebbian learning.

Patterns are vectors of N neurons with values -1 or +1.

Learning (Hebbian rule):
    W_ij = (1/N) * sum over stored patterns mu of  xi_i^mu * xi_j^mu ,   W_ii = 0
    "Neurons that are active together get a positive connection."

Recall (asynchronous updates):
    Repeatedly pick neurons one at a time, in random order, and set
        s_i <- sign( h_i ),   where  h_i = sum_j W_ij s_j   (the "local field")
    If h_i == 0 the neuron keeps its current state.
    Stop when a full sweep over all neurons changes nothing (a fixed point).

Energy:
    E(s) = -1/2 * s^T W s
    With symmetric W and zero diagonal, each asynchronous update can only lower E
    or leave it unchanged, so the network always settles into a fixed point.

All arithmetic is float64. The ±1 patterns may arrive as int8, and int8 sums
over 1024 neurons overflow, so every function converts to float64 first.
"""
import numpy as np


def train_hebbian(patterns):
    """Build the weight matrix from P stored patterns.

    patterns: array of shape (P, N) with values ±1
    returns:  W, float64 array of shape (N, N), symmetric with zero diagonal
    """
    X = np.asarray(patterns, dtype=np.float64)
    if X.ndim == 1:                 # allow a single pattern of shape (N,)
        X = X[None, :]
    N = X.shape[1]
    W = X.T @ X / N                 # sum of outer products xi xi^T, divided by N
    np.fill_diagonal(W, 0.0)        # no self-connections
    return W


def energy(W, state):
    """Hopfield energy E = -1/2 s^T W s."""
    s = np.asarray(state, dtype=np.float64)
    return -0.5 * s @ W @ s


def recall(W, start, rng, max_sweeps=100):
    """Run asynchronous updates from `start` until nothing changes.

    W:          weight matrix (N, N)
    start:      initial state, shape (N,), values ±1 (not modified)
    rng:        numpy random Generator (sets the update order, so results are reproducible)
    max_sweeps: safety limit. One sweep = every neuron updated once.

    returns a dict:
        state     final state (float64, ±1)
        sweeps    number of sweeps run
        converged True if the last sweep changed nothing
        energies  energy at the start and after every sweep
    """
    s = np.asarray(start, dtype=np.float64).copy()
    N = len(s)
    energies = [energy(W, s)]

    for sweep in range(1, max_sweeps + 1):
        n_flips = 0
        for i in rng.permutation(N):          # visit neurons in a random order
            h = W[i] @ s                      # local field of neuron i
            if h != 0 and np.sign(h) != s[i]: # if h == 0, keep the current state
                s[i] = -s[i]
                n_flips += 1
        energies.append(energy(W, s))
        if n_flips == 0:                      # a full sweep with no change = fixed point
            return dict(state=s, sweeps=sweep, converged=True, energies=np.array(energies))

    return dict(state=s, sweeps=max_sweeps, converged=False, energies=np.array(energies))


def is_fixed_point(W, pattern):
    """Fraction of neurons that agree with their own local field (1.0 = pattern is perfectly stable).

    A neuron counts as stable if sign(h_i) == s_i, or h_i == 0 (then the update rule keeps it).
    """
    s = np.asarray(pattern, dtype=np.float64)
    h = W @ s
    stable = (np.sign(h) == s) | (h == 0)
    return stable.mean()


def corrupt(pattern, fraction, rng):
    """Return a copy of `pattern` with exactly round(fraction * N) randomly chosen neurons flipped."""
    s = np.asarray(pattern, dtype=np.float64).copy()
    n_flip = int(round(fraction * len(s)))
    flip = rng.choice(len(s), size=n_flip, replace=False)
    s[flip] = -s[flip]
    return s


def overlap(a, b):
    """Overlap m = (1/N) * sum_i a_i b_i, computed in float64."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a @ b / len(a)
