import unittest
import xmlrunner
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from harmonic_balancer import HarmonicBalancer

class TestImprovedHarmonicBalancer(unittest.TestCase):
    def setUp(self):
        self.balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10)

    def test_run_experiment(self):
        best_solution, best_score = self.balancer.run_experiment()
        self.assertIsNotNone(best_solution)
        self.assertIsInstance(best_score, float)

    def test_generate_new_harmony(self):
        new_harmony = self.balancer.generate_new_harmony(transition_constant=0.1)
        self.assertEqual(len(new_harmony), self.balancer.num_qubits)
        self.assertTrue(np.all(new_harmony >= 0) and np.all(new_harmony <= 1))

    def test_update_harmony_memory(self):
        new_harmony = np.random.rand(self.balancer.num_qubits)
        evolved_state = new_harmony
        score = self.balancer.objective_function(evolved_state, 1)
        self.balancer.update_harmony_memory(new_harmony, evolved_state, score)
        self.assertIn(new_harmony.tolist(), [vec.tolist() for vec in self.balancer.harmony_memory])

    def test_custom_objective_function(self):
        def custom_objective_function(vector, param):
            return np.prod(vector) * param

        balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10, objective_function=custom_objective_function)
        best_solution, best_score = balancer.run_experiment()
        self.assertIsNotNone(best_solution)
        self.assertIsInstance(best_score, float)

    def test_with_pi_constant(self):
        def pi_objective_function(vector, param):
            return np.sum(vector) * np.pi

        balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10, objective_function=pi_objective_function)
        best_solution, best_score = balancer.run_experiment()
        self.assertIsNotNone(best_solution)
        self.assertIsInstance(best_score, float)

    def test_with_exponential_function(self):
        def exp_objective_function(vector, param):
            return np.sum(np.exp(vector))

        balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10, objective_function=exp_objective_function)
        best_solution, best_score = balancer.run_experiment()
        self.assertIsNotNone(best_solution)
        self.assertIsInstance(best_score, float)

    def test_with_sine_function(self):
        def sine_objective_function(vector, param):
            return np.sum(np.sin(vector))

        balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10, objective_function=sine_objective_function)
        best_solution, best_score = balancer.run_experiment()
        self.assertIsNotNone(best_solution)
        self.assertIsInstance(best_score, float)

    def test_with_image_processing_function(self):
        def image_processing_objective_function(vector, param):
            # Simulate an image processing function
            return np.sum(vector) * 255  # Example: scaling pixel values

        balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10, objective_function=image_processing_objective_function)
        best_solution, best_score = balancer.run_experiment()
        self.assertIsNotNone(best_solution)
        self.assertIsInstance(best_score, float)

if __name__ == '__main__':
    with open('tests/test-reports-harmonic-balancer.xml', 'wb' ) as output:
        unittest.main(testRunner=xmlrunner.XMLTestRunner(output=output))