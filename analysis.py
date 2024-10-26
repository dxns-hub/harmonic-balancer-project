import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from ecosystem import EnhancedHumanQuantumEcosystem

def analyze_repeated_shocks(num_simulations=10):
    complexity_histories = []
    
    for _ in tqdm(range(num_simulations), desc="Running simulations"):
        ecosystem = EnhancedHumanQuantumEcosystem(num_groups=5, shock_type='pulse')
        initial_state = np.random.rand(ecosystem.num_groups)
        initial_state /= initial_state.sum()
        
        t_span = (0, 200)
        sol = solve_ivp(ecosystem.ecosystem_dynamics, t_span, initial_state, method='Radau', t_eval=np.linspace(t_span[0], t_span[1], 1000))
        
        complexity_history = np.array([ecosystem.calculate_system_complexity(state) for state in sol.y.T])
        complexity_histories.append(complexity_history)
    
    return sol.t, complexity_histories

def analyze_results(history):
    scores = history['scores']
    states = history['states']
    
    # Example analysis: Calculate the mean and standard deviation of scores
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    return {
        'mean_score': mean_score,
        'std_score': std_score
    }
