import numpy as np

def get_initial_data(N, r, seed=-1):
    """
    Generate N random points in a circle or radius r.
    If seed==-1, don't set a random seed.  Otherwise, set seed to 'seed'.
    """
    if seed != -1:
        np.random.seed(0)

    xs = np.array([])
    ys = np.array([])
    while len(xs) < N:
        x = (np.random.rand(N)*2 - 1)*r
        y = (np.random.rand(N)*2 - 1)*r
        rr = np.hypot(x, y)
        g = rr <= r
        x = x[g]
        y = y[g]
        L = N - len(xs)
        xs = np.concatenate([xs, x[:L]])
        ys = np.concatenate([ys, y[:L]])

    return xs, ys
