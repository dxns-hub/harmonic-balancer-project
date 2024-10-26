import unittest
import xmlrunner
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_reactor_simulation import QuantumReactor

class TestQuantumReactor(unittest.TestCase):
    def setUp(self):
        self.reactor = QuantumReactor(num_groups=5, network_type='small_world', shock_type='pulse', interaction_strength=0.1)

    def test_run_simulation(self):
        states = self.reactor.run_simulation(time_steps=10)
        self.assertEqual(states.shape, (10, 5))

    def test_ecosystem_dynamics_inheritance(self):
        state = np.random.rand(5)
        new_state = self.reactor.ecosystem_dynamics(0, state)
        self.assertEqual(len(new_state), 5)

if __name__ == '__main__': 
    with open('tests/test-reports-quantum-reactor.xml', 'wb') as output:
        unittest.main(testRunner=xmlrunner.XMLTestRunner(output=output))