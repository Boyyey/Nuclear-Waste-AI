import unittest
import torch
import numpy as np
from nuclearai import optimization

class TestOptimization(unittest.TestCase):
    def test_multiobjective_cost(self):
        outputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([0,1])
        objectives = {'obj1': 1, 'obj2': 2}
        cost_fn = optimization.MultiObjectiveCost({'obj1': 1.0, 'obj2': 2.0})
        cost = cost_fn(outputs, mask, objectives)
        self.assertAlmostEqual(cost.item(), 2.0+2*3.0+5.0+2*6.0)

    def test_pareto_front(self):
        points = np.array([[1,2],[2,1],[3,3],[0,4]])
        mask = optimization.pareto_front(points)
        self.assertTrue(np.any(mask))

if __name__ == '__main__':
    unittest.main() 