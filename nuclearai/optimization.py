import torch
import numpy as np

class MultiObjectiveCost:
    def __init__(self, weights):
        self.weights = weights  # dict: objective -> weight

    def __call__(self, outputs, mask, objectives):
        # objectives: dict of name -> column index in outputs
        cost = 0.0
        for obj, idx in objectives.items():
            cost += self.weights.get(obj, 1.0) * outputs[mask, idx].sum()
        return cost

# Example constraint: neutron economy, irradiation time, etc.
def check_constraints(schedule, max_time=1000, max_flux=1e15):
    time_ok = schedule['time'] <= max_time
    flux_ok = np.all(schedule['flux'] <= max_flux)
    return time_ok and flux_ok

# Example genetic algorithm (stub)
def genetic_algorithm(opt_func, population, n_generations=50, mutation_rate=0.1):
    best = None
    best_score = float('inf')
    for gen in range(n_generations):
        scores = [opt_func(ind) for ind in population]
        idx = np.argmin(scores)
        if scores[idx] < best_score:
            best = population[idx]
            best_score = scores[idx]
        # Mutation (simple)
        for i in range(len(population)):
            if np.random.rand() < mutation_rate:
                population[i] += np.random.normal(0, 0.1, size=population[i].shape)
    return best, best_score

# Logging utility
def log_optimization(history, filename="optimization_log.csv"):
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "cost"])
        for step, cost in history:
            writer.writerow([step, cost])

def pareto_front(points):
    """
    Compute the Pareto front for a set of multi-objective points.
    points: np.ndarray of shape (N, M) where M is the number of objectives.
    Returns a boolean mask of points on the Pareto front.
    """
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1) | np.all(points[is_efficient] == c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient 