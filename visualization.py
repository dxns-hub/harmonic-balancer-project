import matplotlib.pyplot as plt
import numpy as np

def plot_complexity_over_time(t_eval, complexity_histories):
    plt.figure(figsize=(12, 6))
    for complexity_history in complexity_histories:
        plt.plot(t_eval, complexity_history, alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('System Complexity')
    plt.title('System Complexity Over Time (Multiple Simulations)')
    plt.savefig('complexity_over_time.png')
    plt.close()

def plot_average_complexity(t_eval, complexity_histories):
    avg_complexity = np.mean(complexity_histories, axis=0)
    std_complexity = np.std(complexity_histories, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(t_eval, avg_complexity, label='Average Complexity')
    plt.fill_between(t_eval, avg_complexity - std_complexity, avg_complexity + std_complexity, alpha=0.3, label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('System Complexity')
    plt.title('Average System Complexity Over Time')
    plt.legend()
    plt.savefig('average_complexity_over_time.png')
    plt.close()

def plot_resilience_distribution(resilience_scores):
    plt.figure(figsize=(10, 6))
    plt.hist(resilience_scores, bins=20)
    plt.xlabel('Resilience Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of System Resilience to Repeated Shocks')
    plt.savefig('resilience_distribution.png')
    plt.close()

def plot_scores(history):
    scores = history['scores']
    plt.plot(scores)
    plt.title('Convergence of Harmonic Balancer Algorithm')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.grid(True)
    plt.show()

def plot_states(history):
    states = history['states']
    plt.plot(states)
    plt.title('States Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('State Values')
    plt.grid(True)
    plt.show()
