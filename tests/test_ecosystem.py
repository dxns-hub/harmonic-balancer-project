import unittest
import xmlrunner
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecosystem import EnhancedHumanQuantumEcosystem

class TestEnhancedHumanQuantumEcosystem(unittest.TestCase):
    def setUp(self):
        self.ecosystem = EnhancedHumanQuantumEcosystem(num_groups=5, network_type='small_world', shock_type='pulse', interaction_strength=0.1)
        self.state = np.random.rand(5)

    def test_generate_network(self):
        network = self.ecosystem.generate_network('small_world')
        self.assertEqual(network.shape, (5, 5))

    def test_ecosystem_dynamics(self):
        new_state = self.ecosystem.ecosystem_dynamics(0, self.state)
        self.assertEqual(len(new_state), 5)

    def test_external_shock(self):
        shock = self.ecosystem.external_shock(0)
        self.assertEqual(len(shock), 5)

if __name__ == '__main__':
    with open('tests/test-reports-ecosystem.xml', 'wb') as output:
        unittest.main(testRunner=xmlrunner.XMLTestRunner(output=output))