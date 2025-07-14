import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np

# --- Graph Construction ---
def build_isotope_graph(isotopes, reactions):
    """
    Build a PyTorch Geometric Data object from isotope and reaction data.
    isotopes: list of dicts with isotope properties
    reactions: list of (from, to, properties) tuples
    """
    G = nx.DiGraph()
    for iso in isotopes:
        G.add_node(iso['id'], **iso)
    for frm, to, props in reactions:
        G.add_edge(frm, to, **props)
    # Node features: [half-life, radiotoxicity, decay heat, cross-section]
    x = []
    for node in G.nodes(data=True):
        d = node[1]
        x.append([
            d.get('half_life', 0),
            d.get('radiotox', 0),
            d.get('decay_heat', 0),
            d.get('cross_section', 0)
        ])
    x = torch.tensor(x, dtype=torch.float)
    # Edge index
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# --- GNN Model ---
class TransmutationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# --- Optimization Routine ---
def optimize_neutron_schedule(model, data, target_mask, n_steps=100, lr=0.05):
    """
    Optimize neutron flux/spectrum schedule to minimize radiotoxicity/heat.
    target_mask: mask for long-lived isotopes to minimize
    """
    # Dummy: treat cross_section as a proxy for neutron schedule
    data.x = data.x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([data.x], lr=lr)
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(data)
        # Cost: sum radiotoxicity and decay heat of target isotopes
        cost = (out[target_mask, 1] + out[target_mask, 2]).sum()
        cost.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: Cost = {cost.item():.4e}")
    return data.x.detach()

# --- MCNP/Serpent Integration Stubs ---
def run_mcnp_simulation(schedule):
    """
    Stub for MCNP integration. Replace with subprocess call or API.
    """
    print("[MCNP] Simulating neutron field with schedule:", schedule)
    # ...
    return np.random.rand()

def run_serpent_simulation(schedule):
    """
    Stub for Serpent integration. Replace with subprocess call or API.
    """
    print("[Serpent] Simulating neutron field with schedule:", schedule)
    # ...
    return np.random.rand()

# --- Example Usage ---
if __name__ == "__main__":
    # Expanded real-world isotopes and reactions (values are illustrative)
    isotopes = [
        {'id': 0, 'name': 'U-238', 'half_life': 4.468e9, 'radiotox': 0.5, 'decay_heat': 0.01, 'cross_section': 2.7},
        {'id': 1, 'name': 'U-235', 'half_life': 7.04e8, 'radiotox': 1.0, 'decay_heat': 0.02, 'cross_section': 680},
        {'id': 2, 'name': 'Pu-239', 'half_life': 2.41e4, 'radiotox': 100, 'decay_heat': 1.9, 'cross_section': 742},
        {'id': 3, 'name': 'Am-241', 'half_life': 432.2, 'radiotox': 120, 'decay_heat': 2.9, 'cross_section': 680},
        {'id': 4, 'name': 'Cs-137', 'half_life': 30.17, 'radiotox': 88, 'decay_heat': 0.6, 'cross_section': 0.1},
        {'id': 5, 'name': 'Sr-90', 'half_life': 28.8, 'radiotox': 28, 'decay_heat': 0.4, 'cross_section': 1.2},
        {'id': 6, 'name': 'Np-237', 'half_life': 2.14e6, 'radiotox': 60, 'decay_heat': 0.5, 'cross_section': 170},
        {'id': 7, 'name': 'Tc-99', 'half_life': 2.11e5, 'radiotox': 6, 'decay_heat': 0.02, 'cross_section': 20},
        {'id': 8, 'name': 'Stable', 'half_life': 0, 'radiotox': 0, 'decay_heat': 0, 'cross_section': 0},
    ]
    reactions = [
        (0, 1, {'type': 'n,gamma'}),  # U-238 -> U-239 (decays to Np-239 -> Pu-239)
        (1, 8, {'type': 'fission'}),  # U-235 -> Stable (fission products)
        (2, 8, {'type': 'fission'}),  # Pu-239 -> Stable (fission products)
        (2, 3, {'type': 'n,gamma'}),  # Pu-239 -> Am-241
        (3, 8, {'type': 'decay'}),    # Am-241 -> Stable (decay chain simplified)
        (4, 8, {'type': 'decay'}),    # Cs-137 -> Stable
        (5, 8, {'type': 'decay'}),    # Sr-90 -> Stable
        (6, 2, {'type': 'n,gamma'}),  # Np-237 -> Pu-239
        (7, 8, {'type': 'decay'}),    # Tc-99 -> Stable
    ]
    data = build_isotope_graph(isotopes, reactions)
    model = TransmutationGNN(in_channels=4, hidden_channels=16, out_channels=4)
    target_mask = torch.tensor([2, 3, 4, 5, 6, 7])  # Minimize long-lived actinides and fission products
    optimized_x = optimize_neutron_schedule(model, data, target_mask)
    # Example MCNP/Serpent call
    run_mcnp_simulation(optimized_x)
    run_serpent_simulation(optimized_x) 