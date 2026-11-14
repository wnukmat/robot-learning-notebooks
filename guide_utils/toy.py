"""A small closed-loop control task used across the notebooks.

A point robot in the unit square must reach a target without touching a round obstacle. An expert controller
supplies demonstrations, and a small network is trained to imitate it and then run on its own.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor

C0 = np.array([0.5, 0.5])   # obstacle centre
R = 0.15                    # obstacle radius
STEP = 0.03                 # largest move per step


def set_obstacle(centre):
    global C0
    C0 = np.asarray(centre, dtype=float)


def expert(p, t):
    """The demonstrator: head for the target, and steer around the obstacle when close to it and heading in."""
    d = t - p; dist = np.linalg.norm(d, axis=-1, keepdims=True); u = d / np.maximum(dist, 1e-9)
    c = p - C0; cd = np.linalg.norm(c, axis=-1, keepdims=True); n = c / np.maximum(cd, 1e-9)
    tang = np.stack([-n[..., 1], n[..., 0]], -1); tang = tang * np.sign(np.sum(tang * u, -1, keepdims=True) + 1e-9)
    w = np.clip((0.30 - cd) / 0.12, 0, 1) * (np.sum(u * -n, -1, keepdims=True) > 0)
    v = (1 - w) * u + w * (tang + 0.6 * n); v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    return v * np.minimum(STEP, dist)


def sample_pairs(rng, n):
    """n random (start, target) pairs that are clear of the obstacle and far enough apart."""
    P, T = [], []
    while len(P) < n:
        p, t = rng.uniform(0.05, 0.95, 2), rng.uniform(0.05, 0.95, 2)
        if min(np.linalg.norm(p - C0), np.linalg.norm(t - C0)) > R + 0.07 and np.linalg.norm(p - t) > 0.45:
            P.append(p); T.append(t)
    return np.array(P), np.array(T)


def features(p, t, kind="relative"):
    """What the policy sees. 'absolute': position and target. 'relative': target and obstacle relative to the robot."""
    return np.concatenate([p, t], -1) if kind == "absolute" else np.concatenate([t - p, C0 - p], -1)


def demonstrations(rng, n, kind="relative", noise=0.0, horizon=1):
    """n expert episodes as (features, actions). noise nudges the expert off course so the data shows recovery.
    horizon > 1 returns the next `horizon` actions for each step, flattened, for chunked policies."""
    P, T = sample_pairs(rng, n); X, Y = [], []
    for p, t in zip(P, T):
        states, acts = [], []
        for _ in range(90):
            a = expert(p, t); states.append(p.copy()); acts.append(a); p = p + a + (rng.normal(0, noise, 2) if noise else 0)
            if np.linalg.norm(p - t) < 0.02:
                break
        acts = np.array(acts + [np.zeros(2)] * horizon)
        for i, s in enumerate(states):
            X.append(features(s[None], t[None], kind)[0]); Y.append(acts[i:i + horizon].ravel())
    return np.array(X), np.array(Y)


class Policy:
    """A small MLP trained by behaviour cloning. Actions are scaled up for training and back down when predicting."""

    def __init__(self, hidden=(64, 64), seed=0, max_iter=1500):
        self.net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter, learning_rate_init=2e-3, tol=1e-7, n_iter_no_change=60, random_state=seed)

    def fit(self, X, Y):
        self.net.fit(X, Y * 30); return self

    def predict(self, X):
        return self.net.predict(X) / 30


def rollout(policy, P, T, kind="relative", steps=110, record=False):
    """Run a policy (or the expert, if policy is None) from each start. Returns the success rate, and paths if record."""
    p = P.copy(); alive = np.ones(len(p), bool); ok = np.zeros(len(p), bool); paths = [p.copy()]
    for _ in range(steps):
        a = policy.predict(features(p, T, kind)) if policy is not None else expert(p, T)
        a = a[:, :2]; n = np.linalg.norm(a, axis=1, keepdims=True); a = a * np.minimum(1, STEP / np.maximum(n, 1e-9))
        p = np.where(alive[:, None], p + a, p); alive &= np.linalg.norm(p - C0, axis=1) > R
        ok |= alive & (np.linalg.norm(p - T, axis=1) < 0.03); alive &= ~ok
        if record:
            paths.append(p.copy())
    return (ok.mean(), np.array(paths), ok) if record else ok.mean()


def test_set(n=300, seed=999):
    return sample_pairs(np.random.default_rng(seed), n)
