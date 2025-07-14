# nuclearai Documentation

## Overview

`nuclearai` is a modular, research-grade Python package for AI-guided nuclear waste transmutation using graph neural networks (GNNs), advanced optimization, and Monte Carlo code integration.

---

## Modules

### `data`
- `Isotope`, `NuclearReaction`, `NuclearDatabase`: Classes for representing isotopes, reactions, and nuclear networks.
- `load_from_endf(endf_file)`: Load nuclear data from ENDF/B files (requires `openmc`).

### `gnn`
- `TransmutationGNN`: Flexible GNN model for nuclear transmutation graphs.

### `physics`
- `get_decay_chain`, `build_full_decay_chains`: Utilities for decay/activation chain analysis.
- `interpolate_cross_section`, `maxwellian_spectrum`, `fission_spectrum`: Physics utilities.

### `optimization`
- `MultiObjectiveCost`: Multi-objective cost function for optimization.
- `genetic_algorithm`, `pareto_front`, `log_optimization`: Optimization and analysis tools.

### `mc_integration`
- `run_mcnp`, `run_serpent`, `parse_mcnp_output`, `parse_serpent_output`: MCNP/Serpent integration.

### `visualization`
- `plot_isotope_network`, `plot_time_evolution`, `plot_optimization_progress`, `plot_sankey_flows`, `plot_interactive_network`, `animate_time_evolution`: Visualization tools.

### `cli`
- Command-line interface for running experiments, loading scenarios, and exporting results.

---

## Usage Examples

See `examples.py` for end-to-end usage.

---

## Advanced Features
- Real nuclear data parsing (ENDF/B, JEFF)
- Full decay/activation chain modeling
- Multi-objective optimization and Pareto analysis
- Batch scenario support (YAML/JSON)
- Advanced visualizations (Sankey, interactive, animation)
- MCNP/Serpent feedback loop

---

## API Reference

(See docstrings in each module for details) 