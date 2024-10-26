# quantum_reactor_simulation.py

import numpy as np
from ecosystem import EnhancedHumanQuantumEcosystem

class QuantumReactor(EnhancedHumanQuantumEcosystem):
    def __init__(self, num_groups=5, network_type='small_world', shock_type='pulse', interaction_strength=0.1):
        super().__init__(num_groups, network_type, shock_type, interaction_strength)
        self.reactor_state = np.random.rand(num_groups)

    def run_simulation(self, time_steps=200):
        states = []
        for t in range(time_steps):
            self.reactor_state = self.ecosystem_dynamics(t, self.reactor_state)
            states.append(self.reactor_state.copy())
        return np.array(states)

def run_quantum_reactor_simulation():
    reactor = QuantumReactor(num_groups=5, network_type='small_world', shock_type='pulse', interaction_strength=0.1)
    states = reactor.run_simulation(time_steps=200)
    
    for t, state in enumerate(states):
        print(f"State at time {t}: {state}")

if __name__ == "__main__":
    run_quantum_reactor_simulation()
