import numpy as np
import networkx as nx

class Isotope:
    def __init__(self, name, Z, A, half_life, radiotox, decay_heat, cross_sections):
        self.name = name
        self.Z = Z  # Atomic number
        self.A = A  # Mass number
        self.half_life = half_life
        self.radiotox = radiotox
        self.decay_heat = decay_heat
        self.cross_sections = cross_sections  # dict: reaction type -> value

    def __repr__(self):
        return f"{self.name} (Z={self.Z}, A={self.A})"

class NuclearReaction:
    def __init__(self, from_iso, to_iso, reaction_type, branching_ratio=1.0):
        self.from_iso = from_iso
        self.to_iso = to_iso
        self.reaction_type = reaction_type
        self.branching_ratio = branching_ratio

    def __repr__(self):
        return f"{self.from_iso} --{self.reaction_type}({self.branching_ratio})--> {self.to_iso}"

class NuclearDatabase:
    def __init__(self):
        self.isotopes = {}
        self.reactions = []

    def add_isotope(self, iso: Isotope):
        self.isotopes[iso.name] = iso

    def add_reaction(self, reaction: NuclearReaction):
        self.reactions.append(reaction)

    def build_graph(self):
        G = nx.DiGraph()
        for iso in self.isotopes.values():
            G.add_node(iso.name, **iso.__dict__)
        for rxn in self.reactions:
            G.add_edge(rxn.from_iso, rxn.to_iso, reaction_type=rxn.reaction_type, branching_ratio=rxn.branching_ratio)
        return G

    @staticmethod
    def from_dicts(isotope_dicts, reaction_dicts):
        db = NuclearDatabase()
        for iso in isotope_dicts:
            db.add_isotope(Isotope(**iso))
        for rxn in reaction_dicts:
            db.add_reaction(NuclearReaction(**rxn))
        return db

def load_from_endf(endf_file):
    """
    Load isotope and reaction data from an ENDF/B nuclear data file using openmc.data.
    Extracts cross-sections, decay data, and builds a NuclearDatabase.
    """
    try:
        import openmc.data as omcdata  # type: ignore
    except ImportError:
        raise ImportError("openmc is required for ENDF/B parsing. Please install openmc.")
    db = NuclearDatabase()
    try:
        nuclide = omcdata.IncidentNeutron.from_hdf5(endf_file)
        iso = Isotope(
            name=nuclide.name,
            Z=nuclide.atomic_number,
            A=nuclide.mass_number,
            half_life=getattr(nuclide, 'half_life', 0),
            radiotox=0,  # Placeholder, real value from radiological data
            decay_heat=0,  # Placeholder
            cross_sections={rxn: nuclide.reactions[rxn].xs['0K'] for rxn in nuclide.reactions}
        )
        db.add_isotope(iso)
        # Add reactions
        for rxn in nuclide.reactions:
            rxn_type = nuclide.reactions[rxn].mt
            # For now, just add self-reaction (stub)
            db.add_reaction(NuclearReaction(iso.name, iso.name, str(rxn_type)))
    except Exception as e:
        print(f"Error loading ENDF/B file {endf_file}: {e}")
    return db 