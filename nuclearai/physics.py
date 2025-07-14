import numpy as np

# Example decay chain utility
def get_decay_chain(isotope, database, max_depth=10):
    chain = [isotope]
    current = isotope
    for _ in range(max_depth):
        found = False
        for rxn in database.reactions:
            if rxn.from_iso == current and rxn.reaction_type == 'decay':
                chain.append(rxn.to_iso)
                current = rxn.to_iso
                found = True
                break
        if not found:
            break
    return chain

# Example cross-section interpolation
def interpolate_cross_section(isotope, reaction_type, energy_grid, cross_section_table):
    # cross_section_table: dict of (isotope, reaction_type) -> (energies, values)
    key = (isotope, reaction_type)
    if key not in cross_section_table:
        return np.zeros_like(energy_grid)
    energies, values = cross_section_table[key]
    return np.interp(energy_grid, energies, values)

# Example neutron spectrum models
def maxwellian_spectrum(energy_grid, temperature_keV=0.025):
    kT = temperature_keV
    spectrum = energy_grid * np.exp(-energy_grid / kT)
    return spectrum / np.trapz(spectrum, energy_grid)

def fission_spectrum(energy_grid):
    # Watt spectrum for fission neutrons
    a, b = 0.988, 2.249
    spectrum = np.exp(-energy_grid / a) * np.sinh(np.sqrt(b * energy_grid))
    return spectrum / np.trapz(spectrum, energy_grid)

def build_full_decay_chains(database, start_isotopes=None, max_depth=20):
    """
    Build all possible decay and activation chains from the given NuclearDatabase.
    Returns a dict: {start_isotope: [chain1, chain2, ...]}
    """
    if start_isotopes is None:
        start_isotopes = list(database.isotopes.keys())
    chains = {}
    for iso in start_isotopes:
        chains[iso] = []
        stack = [([iso], iso)]
        while stack:
            path, current = stack.pop()
            found = False
            for rxn in database.reactions:
                if rxn.from_iso == current and rxn.to_iso != current:
                    new_path = path + [rxn.to_iso]
                    if len(new_path) > max_depth:
                        continue
                    chains[iso].append(new_path)
                    stack.append((new_path, rxn.to_iso))
                    found = True
            if not found and len(path) > 1:
                chains[iso].append(path)
    return chains 