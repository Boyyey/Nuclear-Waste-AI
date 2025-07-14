"""
Example usage of the nuclearai package for AI-guided nuclear waste transmutation.
"""
from nuclearai import data, gnn, physics, optimization, mc_integration, visualization
import torch
import numpy as np

# Example: Load nuclear data (stub or real ENDF/B file)
db = data.NuclearDatabase()
db.add_isotope(data.Isotope('U-238', 92, 238, 4.468e9, 0.5, 0.01, {'n,gamma': 2.7}))
db.add_isotope(data.Isotope('Pu-239', 94, 239, 2.41e4, 100, 1.9, {'fission': 742}))
db.add_reaction(data.NuclearReaction('U-238', 'Pu-239', 'n,gamma'))
G = db.build_graph()

# Visualize the isotope network
visualization.plot_isotope_network(G, highlight_nodes=['Pu-239'])

# Build GNN input
data_x = torch.tensor([[4.468e9, 0.5, 0.01, 2.7], [2.41e4, 100, 1.9, 742]], dtype=torch.float)
edge_index = torch.tensor([[0, 1]], dtype=torch.long).t()  # shape [2, num_edges]
from torch_geometric.data import Data as GeoData
geo_data = GeoData(x=data_x, edge_index=edge_index)

# Run GNN
model = gnn.TransmutationGNN(in_channels=4, hidden_channels=8, out_channels=4)
out = model(geo_data)
print("GNN output:", out)

# Optimization
cost_fn = optimization.MultiObjectiveCost({'radiotox': 1.0, 'decay_heat': 0.5})
mask = torch.tensor([1])  # Minimize Pu-239
objectives = {'radiotox': 1, 'decay_heat': 2}
cost = cost_fn(out, mask, objectives)
print("Optimization cost:", float(cost) if hasattr(cost, 'item') else cost)

# MCNP/Serpent integration (stub)
# mc_integration.run_mcnp('input.i', 'output.o')

# Advanced visualization
visualization.plot_sankey_flows([100, -80, -20], ['U-238', 'Pu-239', 'Fission Products'])
visualization.plot_interactive_network(G)
visualization.animate_time_evolution(np.arange(10), np.random.rand(10,2), ['U-238', 'Pu-239']) 