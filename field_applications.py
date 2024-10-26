import numpy as np
from harmonic_balancer_project.enhanced_ecosystem_with_psi_and_frequency import EnhancedHumanQuantumEcosystem
from harmonic_balancer import HarmonicBalancer

def science_physics_example(ecosystem, time):
    # Example: Calculating the energy of a photon using Planck's constant
    frequency = ecosystem.base_frequency
    energy = ecosystem.constants['planck'] * frequency
    return f"Energy of a photon at {frequency} Hz: {energy} J"

def mathematics_geometry_example(ecosystem):
    # Example: Calculating the area of a circle using Pi
    radius = 5
    area = ecosystem.constants['pi'] * radius**2
    return f"Area of a circle with radius {radius}: {area} square units"

def engineering_electrical_example(ecosystem, voltage):
    # Example: Calculating current using Ohm's law
    resistance = 100  # ohms
    current = voltage / resistance
    return f"Current in a {resistance} ohm resistor at {voltage} volts: {current} amperes"

def healthcare_epidemiology_example(ecosystem, initial_infected, days):
    # Example: Simple SIR model for disease spread
    total_population = 1000000
    recovery_rate = 0.1
    transmission_rate = 0.3
    
    susceptible = total_population - initial_infected
    infected = initial_infected
    recovered = 0
    
    for _ in range(days):
        new_infections = (transmission_rate * infected * susceptible) / total_population
        new_recoveries = recovery_rate * infected
        susceptible -= new_infections
        infected += new_infections - new_recoveries
        recovered += new_recoveries
    
    return f"After {days} days: Susceptible: {susceptible:.0f}, Infected: {infected:.0f}, Recovered: {recovered:.0f}"

def atmospheric_meteorology_example(ecosystem, temperature_celsius):
    # Example: Converting Celsius to Kelvin
    temperature_kelvin = temperature_celsius + 273.15
    return f"{temperature_celsius}°C is equal to {temperature_kelvin}K"

def geological_seismology_example(ecosystem, distance, time):
    # Example: Calculating seismic wave velocity
    velocity = distance / time
    return f"Seismic wave velocity: {velocity} m/s"

def environmental_ecology_example(ecosystem, initial_population, carrying_capacity, growth_rate, time):
    # Example: Logistic growth model
    population = initial_population * ecosystem.constants['e']**(growth_rate * time) / (1 + initial_population * (ecosystem.constants['e']**(growth_rate * time) - 1) / carrying_capacity)
    return f"Population after {time} time units: {population:.2f}"

def logistic_growth_example():
    def logistic_growth_objective(vector, param):
        r = 0.1  # Growth rate
        K = 100  # Carrying capacity
        return np.sum(vector) * r * (1 - np.sum(vector) / K)

    balancer = HarmonicBalancer(num_qubits=4, max_iterations=100, harmony_memory_size=10, objective_function=logistic_growth_objective)
    best_solution, best_score = balancer.run_experiment()

    print("Best solution:", best_solution)
    print("Best score:", best_score)

def run_examples():
    ecosystem = EnhancedHumanQuantumEcosystem()
    
    print(science_physics_example(ecosystem, 5e14))
    print(mathematics_geometry_example(ecosystem))
    print(engineering_electrical_example(ecosystem, 220))
    print(healthcare_epidemiology_example(ecosystem, 100, 30))
    print(atmospheric_meteorology_example(ecosystem, 25))
    print(geological_seismology_example(ecosystem, 1000, 2))
    print(environmental_ecology_example(ecosystem, 100, 1000, 0.1, 50))

if __name__ == "__main__":
    run_examples()
    logistic_growth_example()
