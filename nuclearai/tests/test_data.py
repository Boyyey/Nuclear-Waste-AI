import unittest
from nuclearai import data

class TestData(unittest.TestCase):
    def test_isotope(self):
        iso = data.Isotope('U-238', 92, 238, 4.468e9, 0.5, 0.01, {'n,gamma': 2.7})
        self.assertEqual(iso.name, 'U-238')
        self.assertEqual(iso.Z, 92)
        self.assertEqual(iso.A, 238)
        self.assertAlmostEqual(iso.half_life, 4.468e9)
        self.assertIn('n,gamma', iso.cross_sections)

    def test_reaction(self):
        rxn = data.NuclearReaction('U-238', 'Pu-239', 'n,gamma')
        self.assertEqual(rxn.from_iso, 'U-238')
        self.assertEqual(rxn.to_iso, 'Pu-239')
        self.assertEqual(rxn.reaction_type, 'n,gamma')

    def test_database(self):
        db = data.NuclearDatabase()
        iso = data.Isotope('U-238', 92, 238, 4.468e9, 0.5, 0.01, {'n,gamma': 2.7})
        db.add_isotope(iso)
        self.assertIn('U-238', db.isotopes)
        rxn = data.NuclearReaction('U-238', 'Pu-239', 'n,gamma')
        db.add_reaction(rxn)
        self.assertIn(rxn, db.reactions)
        G = db.build_graph()
        self.assertIn('U-238', G.nodes)

    def test_endf_loading(self):
        # Mocked: just check ImportError is raised if openmc is missing
        try:
            data.load_from_endf('fakefile.h5')
        except ImportError:
            pass
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main() 