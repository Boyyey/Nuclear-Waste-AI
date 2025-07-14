import argparse
from nuclearai import data, gnn, physics, optimization, mc_integration, visualization
import json
import yaml
import os

def load_scenario(scenario_file):
    """
    Load a scenario from a YAML or JSON file.
    """
    if scenario_file.endswith('.yaml') or scenario_file.endswith('.yml'):
        with open(scenario_file) as f:
            return yaml.safe_load(f)
    elif scenario_file.endswith('.json'):
        with open(scenario_file) as f:
            return json.load(f)
    else:
        raise ValueError('Unsupported scenario file format')

def main():
    parser = argparse.ArgumentParser(description='AI-Guided Nuclear Waste Transmutation Network')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Run scenario
    parser_run = subparsers.add_parser('run', help='Run a single scenario')
    parser_run.add_argument('--scenario', type=str, default='default', help='Scenario name or config file')
    parser_run.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat'], help='GNN model type')
    parser_run.add_argument('--opt', type=str, default='adam', choices=['adam', 'genetic'], help='Optimization algorithm')
    parser_run.add_argument('--plot', action='store_true', help='Plot results')
    parser_run.add_argument('--run-mc', action='store_true', help='Run MCNP/Serpent integration')

    # Batch
    parser_batch = subparsers.add_parser('batch', help='Run batch experiments')
    parser_batch.add_argument('--scenario-dir', type=str, default='nuclearai/scenario_templates', help='Directory with scenarios')
    parser_batch.add_argument('--output-dir', type=str, default='batch_results', help='Output directory')

    # Report
    parser_report = subparsers.add_parser('report', help='Generate auto-report for batch results')
    parser_report.add_argument('--results-dir', type=str, default='batch_results', help='Results directory')
    parser_report.add_argument('--output-file', type=str, default='auto_report.html', help='Output HTML file')

    # RL
    parser_rl = subparsers.add_parser('rl', help='Run RL optimization (stub)')
    parser_rl.add_argument('--scenario', type=str, default='default', help='Scenario name or config file')

    # Bayesian optimization
    parser_bayes = subparsers.add_parser('bayes', help='Run Bayesian optimization (stub)')
    parser_bayes.add_argument('--scenario', type=str, default='default', help='Scenario name or config file')

    args = parser.parse_args()

    if args.command == 'run':
        if os.path.exists(args.scenario):
            scenario = load_scenario(args.scenario)
            print(f"Loaded scenario: {scenario}")
        else:
            scenario = None
            print(f"Using built-in scenario: {args.scenario}")

        # Build graph, model, etc. (stub)
        print(f"Using model: {args.model}")
        # ...build and run model...

        # Optimization (stub)
        print(f"Optimization algorithm: {args.opt}")
        # ...run optimization...

        # MCNP/Serpent integration (stub)
        if args.run_mc:
            print("Running MCNP/Serpent integration...")
            # ...call integration...

        # Plotting
        if args.plot:
            print("Plotting results...")
            # ...call visualization...
    elif args.command == 'batch':
        from nuclearai.batch_runner import run_batch
        run_batch(args.scenario_dir, args.output_dir)
    elif args.command == 'report':
        from nuclearai.auto_report import generate_report
        generate_report(args.results_dir, args.output_file)
    elif args.command == 'rl':
        print('RL optimization not yet fully implemented.')
    elif args.command == 'bayes':
        print('Bayesian optimization not yet fully implemented.')
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 