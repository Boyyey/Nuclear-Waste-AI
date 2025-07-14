import unittest
import torch
from nuclearai import gnn
from torch_geometric.data import Data as GeoData

class TestGNN(unittest.TestCase):
    def test_gnn_forward(self):
        x = torch.rand(5, 4)
        edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
        data = GeoData(x=x, edge_index=edge_index)
        model = gnn.TransmutationGNN(in_channels=4, hidden_channels=8, out_channels=4)
        out = model(data)
        self.assertEqual(out.shape, (5,4))

if __name__ == '__main__':
    unittest.main() 