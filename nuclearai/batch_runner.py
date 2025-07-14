"""
Batch experiment runner for nuclearai.
Loads scenarios, runs optimization, logs and exports results.
"""
import os
import glob
import json
import yaml
from nuclearai import data, gnn, optimization, visualization
import torch
import numpy as np

def run_batch(scenario_dir, output_dir):
    """
    Run batch experiments for all scenario files in scenario_dir.
    Save results and logs to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    scenario_files = glob.glob(os.path.join(scenario_dir, '*.yaml')) + glob.glob(os.path.join(scenario_dir, '*.json'))
    summary = []
    for scen_file in scenario_files:
        print(f"Running scenario: {scen_file}")
        if scen_file.endswith('.yaml') or scen_file.endswith('.yml'):
            with open(scen_file) as f:
                scenario = yaml.safe_load(f)
        else:
            with open(scen_file) as f:
                scenario = json.load(f)
        # Build database
        db = data.NuclearDatabase.from_dicts(scenario['isotopes'], scenario['reactions'])
        G = db.build_graph()
        # GNN input
        x = []
        for iso in scenario['isotopes']:
            x.append([iso['half_life'], iso['radiotox'], iso['decay_heat'], list(iso['cross_sections'].values())[0]])
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        from torch_geometric.data import Data as GeoData
        geo_data = GeoData(x=x, edge_index=edge_index)
        # Run GNN
        model = gnn.TransmutationGNN(in_channels=4, hidden_channels=8, out_channels=4)
        out = model(geo_data)
        # Optimization
        cost_fn = optimization.MultiObjectiveCost({'radiotox': 1.0, 'decay_heat': 0.5})
        mask = torch.arange(len(scenario['isotopes']))
        objectives = {'radiotox': 1, 'decay_heat': 2}
        cost = cost_fn(out, mask, objectives)
        # Log
        result = {'scenario': os.path.basename(scen_file), 'cost': float(cost) if hasattr(cost, 'item') else cost}
        summary.append(result)
        with open(os.path.join(output_dir, os.path.basename(scen_file) + '.result.json'), 'w') as f:
            json.dump(result, f, indent=2)
    # Export summary
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Batch run complete. Results in {output_dir}") 