"""A 6 x 9 grid world with walls, a goal (+1) and a hazard (-1), for the reinforcement learning notebook."""
import numpy as np

H, W = 6, 9
WALLS = {(1, 2), (2, 2), (3, 2), (3, 3), (1, 5), (2, 5), (4, 5), (4, 6), (1, 7)}
GOAL, PIT = (1, 8), (3, 6)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
FREE = [(r, c) for r in range(H) for c in range(W) if (r, c) not in WALLS]
STARTS = [s for s in FREE if s not in (GOAL, PIT)]
SLIP, GAMMA, STEP_COST = 0.1, 0.95, -0.02


def move(s, a):
    r, c = s[0] + a[0], s[1] + a[1]
    return s if (r, c) in WALLS or not (0 <= r < H and 0 <= c < W) else (r, c)


def reward(s2):
    return 1.0 if s2 == GOAL else (-1.0 if s2 == PIT else STEP_COST)


def outcomes(a):
    """The action taken with probability 1 - SLIP, and each sideways slip with probability SLIP / 2."""
    side = [b for b in ACTIONS if b != a and b != (-a[0], -a[1])]
    return [(a, 1 - SLIP)] + [(b, SLIP / 2) for b in side]


def step(s, i, rng):
    """Take action index i from state s. Returns (next state, reward, done)."""
    a = ACTIONS[i]
    if rng.random() < SLIP:
        side = [b for b in ACTIONS if b != a and b != (-a[0], -a[1])]; a = side[rng.integers(2)]
    s2 = move(s, a)
    return s2, reward(s2), s2 in (GOAL, PIT)


def value_iteration(sweeps=400):
    V = {s: 0.0 for s in FREE}
    q = lambda s, a: sum(p * (reward(move(s, b)) + GAMMA * V[move(s, b)]) for b, p in outcomes(a))
    for _ in range(sweeps):
        for s in STARTS:
            V[s] = max(q(s, a) for a in ACTIONS)
    return V, {s: max(range(4), key=lambda i: q(s, ACTIONS[i])) for s in FREE}


def evaluate(policy_fn, rng, repeats=3, max_steps=60):
    """Success rate of policy_fn(state) -> action index, from every start."""
    wins = 0
    for s0 in STARTS:
        for _ in range(repeats):
            s = s0
            for _ in range(max_steps):
                s, _r, done = step(s, policy_fn(s), rng)
                if done:
                    break
            wins += s == GOAL
    return wins / (len(STARTS) * repeats)
