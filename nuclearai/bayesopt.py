import numpy as np

def bayesian_optimize(objective_func, param_space, n_calls=30):
    """
    Run Bayesian optimization on the given objective function.
    param_space: list of skopt.space dimensions (see skopt docs)
    Returns the best parameters and the optimization result.
    """
    try:
        from skopt import gp_minimize  # type: ignore
    except ImportError:
        raise ImportError("scikit-optimize (skopt) is required for Bayesian optimization. Please install skopt.")
    res = gp_minimize(objective_func, param_space, n_calls=n_calls, random_state=0)
    return res.x, res 