import numpy as np
import networkx as nx

class EnhancedHumanQuantumEcosystem:
    def __init__(self, num_groups=5, network_type='small_world', shock_type='pulse', interaction_strength=0.1):
        self.num_groups = num_groups
        self.growth_rates = np.random.uniform(0.01, 0.05, num_groups)
        self.carrying_capacities = np.random.uniform(0.5, 1.0, num_groups)
        self.innovation_rates = np.random.uniform(0.001, 0.01, num_groups)
        self.adaptation_rates = np.random.uniform(0.01, 0.05, num_groups)
        self.interaction_matrix = self.generate_network(network_type) * interaction_strength
        self.shock_type = shock_type
        self.shock_history = []

    def generate_network(self, network_type):
        if network_type == 'small_world':
            G = nx.watts_strogatz_graph(self.num_groups, 2, 0.3)
        elif network_type == 'scale_free':
            G = nx.barabasi_albert_graph(self.num_groups, 1)
        elif network_type == 'random':
            G = nx.erdos_renyi_graph(self.num_groups, 0.5)
        return nx.to_numpy_array(G)

    def ecosystem_dynamics(self, t, state):
        state = np.clip(state, 1e-10, 1)
        state /= np.sum(state)
        
        growth = self.growth_rates * state * (1 - state / self.carrying_capacities)
        interactions = np.dot(self.interaction_matrix, state) - np.dot(state, self.interaction_matrix)
        innovation = self.innovation_rates * np.sqrt(state)
        adaptation = self.adaptation_rates * (1 - state)
        external_shock = self.external_shock(t)
        
        return growth + interactions + innovation + adaptation + external_shock

    def external_shock(self, t):
        shock = np.zeros(self.num_groups)
        if self.shock_type == 'pulse':
            if 50 <= t <= 55 or 100 <= t <= 105 or 150 <= t <= 155:
                shock = -0.05 * np.random.rand(self.num_groups)
        elif self.shock_type == 'sine':
            shock = 0.025 * np.sin(0.1 * t) * np.random.rand(self.num_groups)
        elif self.shock_type == 'step':
            if t >= 100:
                shock = -0.025 * np.ones(self.num_groups)
        elif self.shock_type == 'complex':
            shock = 0.025 * np.sin(0.1 * t) * np.random.rand(self.num_groups)
            if 50 <= t <= 55 or 150 <= t <= 155:
                shock -= 0.05 * np.random.rand(self.num_groups)
            if t >= 100:
                shock -= 0.015 * np.ones(self.num_groups)
        self.shock_history.append(np.mean(shock))
        return shock

    def calculate_system_complexity(self, state):
        state = np.clip(state, 1e-10, 1)
        state /= np.sum(state)
        entropy = -np.sum(state * np.log2(state))
        return entropy / np.log2(self.num_groups)

if __name__ == "__main__":
    ecosystem = EnhancedHumanQuantumEcosystem(num_groups=5, network_type='small_world', shock_type='pulse', interaction_strength=0.1)
    state = np.random.rand(5)
    for t in range(200):
        state = ecosystem.ecosystem_dynamics(t, state)
        print(f"State at time {t}: {state}")
