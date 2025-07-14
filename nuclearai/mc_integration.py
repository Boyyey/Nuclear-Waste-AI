import subprocess
import os

def run_mcnp(input_file, output_file, mcnp_path='mcnp6'):  # mcnp_path: path to MCNP executable
    cmd = [mcnp_path, 'i=' + input_file, 'o=' + output_file]
    print(f"Running MCNP: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("MCNP error:", result.stderr)
    return output_file

def run_serpent(input_file, output_file, serpent_path='sss2'):  # serpent_path: path to Serpent executable
    cmd = [serpent_path, input_file, '-omp', '4']
    print(f"Running Serpent: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Serpent error:", result.stderr)
    return output_file

def parse_mcnp_output(output_file):
    # Stub: parse MCNP output for neutron flux, reaction rates, etc.
    with open(output_file) as f:
        lines = f.readlines()
    # ...parse lines...
    return {'flux': 1e14, 'reaction_rates': {}}

def parse_serpent_output(output_file):
    # Stub: parse Serpent output for neutron flux, reaction rates, etc.
    with open(output_file) as f:
        lines = f.readlines()
    # ...parse lines...
    return {'flux': 1e14, 'reaction_rates': {}} 