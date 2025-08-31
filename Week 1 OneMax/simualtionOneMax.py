# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.patches as mpatches
# from deap import base, creator, tools
# import random

# # Set up the problem and algorithm parameters
# ONE_MAX_LENGTH = 30  # Shorter for better visualization
# POPULATION_SIZE = 50
# P_CROSSOVER = 0.7
# P_MUTATION = 0.1  # Slightly higher for more visible changes
# MAX_GENERATIONS = 20

# # Set random seed for reproducibility
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# # Create a custom colormap from blue (low fitness) to red (high fitness)
# colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.2)]  # Blue to Red
# cmap = LinearSegmentedColormap.from_list('fitness_cmap', colors, N=100)

# # Initialize DEAP
# toolbox = base.Toolbox()
# toolbox.register("zeroOrOne", random.randint, 0, 1)
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
# toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)
# toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# def oneMaxFitness(individual):
#     return sum(individual),

# toolbox.register("evaluate", oneMaxFitness)
# toolbox.register("select", tools.selRoulette)
# toolbox.register("mate", tools.cxOnePoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

# # Create the figure and axis
# plt.style.use('dark_background')
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
# fig.suptitle('Genetic Algorithm Evolution Simulation', fontsize=16, fontweight='bold', color='cyan')

# # Initialize data storage for visualization
# generation_data = []
# best_individuals = []
# avg_fitness = []
# max_fitness = []

# # Create initial population
# population = toolbox.populationCreator(n=POPULATION_SIZE)
# fitnessValues = list(map(toolbox.evaluate, population))
# for individual, fitnessValue in zip(population, fitnessValues):
#     individual.fitness.values = fitnessValue

# # Set up the visualization
# def init_visualization():
#     ax1.clear()
#     ax2.clear()
    
#     # Configure the first subplot (population visualization)
#     ax1.set_xlim(0, ONE_MAX_LENGTH)
#     ax1.set_ylim(0, POPULATION_SIZE)
#     ax1.set_xlabel('Gene Position', fontweight='bold')
#     ax1.set_ylabel('Individual in Population', fontweight='bold')
#     ax1.set_title('Population Genome (0=blue, 1=red)', fontweight='bold')
#     ax1.grid(True, alpha=0.3)
    
#     # Configure the second subplot (fitness progression)
#     ax2.set_xlim(0, MAX_GENERATIONS)
#     ax2.set_ylim(0, ONE_MAX_LENGTH)
#     ax2.set_xlabel('Generation', fontweight='bold')
#     ax2.set_ylabel('Fitness', fontweight='bold')
#     ax2.set_title('Fitness Progression', fontweight='bold')
#     ax2.grid(True, alpha=0.3)
    
#     # Add legend
#     blue_patch = mpatches.Patch(color='blue', label='0 (Off)')
#     red_patch = mpatches.Patch(color='red', label='1 (On)')
#     ax1.legend(handles=[blue_patch, red_patch], loc='upper right')
    
#     return fig,

# # Update function for animation
# def update(generation):
#     global population
    
#     ax1.clear()
#     ax2.clear()
    
#     # Plot the current population
#     for i, individual in enumerate(population):
#         for j, gene in enumerate(individual):
#             color = 'red' if gene == 1 else 'blue'
#             ax1.scatter(j, i, color=color, s=30, alpha=0.7)
    
#     # Calculate and store fitness statistics
#     fitness_values = [ind.fitness.values[0] for ind in population]
#     current_avg = sum(fitness_values) / len(population)
#     current_max = max(fitness_values)
#     avg_fitness.append(current_avg)
#     max_fitness.append(current_max)
    
#     # Store the best individual
#     best_index = fitness_values.index(current_max)
#     best_individuals.append(population[best_index])
    
#     # Plot fitness progression
#     generations = range(len(avg_fitness))
#     ax2.plot(generations, avg_fitness, 'o-', color='cyan', label='Average Fitness', linewidth=2, markersize=4)
#     ax2.plot(generations, max_fitness, 'o-', color='yellow', label='Max Fitness', linewidth=2, markersize=4)
    
#     # Configure the first subplot
#     ax1.set_xlim(0, ONE_MAX_LENGTH)
#     ax1.set_ylim(0, POPULATION_SIZE)
#     ax1.set_xlabel('Gene Position', fontweight='bold')
#     ax1.set_ylabel('Individual in Population', fontweight='bold')
#     ax1.set_title(f'Generation {generation}: Population Genome', fontweight='bold')
#     ax1.grid(True, alpha=0.3)
    
#     # Configure the second subplot
#     ax2.set_xlim(0, MAX_GENERATIONS)
#     ax2.set_ylim(0, ONE_MAX_LENGTH)
#     ax2.set_xlabel('Generation', fontweight='bold')
#     ax2.set_ylabel('Fitness', fontweight='bold')
#     ax2.set_title('Fitness Progression', fontweight='bold')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend(loc='lower right')
    
#     # Add text annotations
#     ax2.text(0.02, 0.98, f'Max Fitness: {current_max}/{ONE_MAX_LENGTH}', 
#              transform=ax2.transAxes, verticalalignment='top', 
#              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
#     ax2.text(0.02, 0.88, f'Avg Fitness: {current_avg:.2f}', 
#              transform=ax2.transAxes, verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.5))
    
#     # Add legend to first subplot
#     blue_patch = mpatches.Patch(color='blue', label='0 (Off)')
#     red_patch = mpatches.Patch(color='red', label='1 (On)')
#     ax1.legend(handles=[blue_patch, red_patch], loc='upper right')
    
#     # Stop if we've reached the maximum generations
#     if generation >= MAX_GENERATIONS or current_max == ONE_MAX_LENGTH:
#         ani.event_source.stop()
#         plt.tight_layout()
#         return fig,
    
#     # Evolutionary algorithm steps
#     # Selection
#     offspring = toolbox.select(population, len(population))
#     offspring = list(map(toolbox.clone, offspring))
    
#     # Crossover
#     for child1, child2 in zip(offspring[::2], offspring[1::2]):
#         if random.random() < P_CROSSOVER:
#             toolbox.mate(child1, child2)
#             del child1.fitness.values
#             del child2.fitness.values

#     # Mutation
#     for mutant in offspring:
#         if random.random() < P_MUTATION:
#             toolbox.mutate(mutant)
#             del mutant.fitness.values

#     # Evaluate fitness for new individuals
#     freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
#     freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
#     for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
#         individual.fitness.values = fitnessValue

#     # Replace population
#     population[:] = offspring
    
#     return fig,

# # Create animation
# ani = FuncAnimation(fig, update, frames=MAX_GENERATIONS+1,
#                     init_func=init_visualization, interval=800, repeat=False)

# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from deap import base, creator, tools
import random

# Set up the problem and algorithm parameters
ONE_MAX_LENGTH = 30  # Shorter for better visualization
POPULATION_SIZE = 50
P_CROSSOVER = 0.7
P_MUTATION = 0.1  # Slightly higher for more visible changes
MAX_GENERATIONS = 20

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create a custom colormap from blue (low fitness) to red (high fitness)
colors = [(0.2, 0.4, 0.8), (0.8, 0.2, 0.2)]  # Blue to Red
cmap = LinearSegmentedColormap.from_list('fitness_cmap', colors, N=100)

# Initialize DEAP
toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def oneMaxFitness(individual):
    return sum(individual),

toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

# Create the figure and axis
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Genetic Algorithm Evolution Simulation', fontsize=16, fontweight='bold', color='cyan')

# Initialize data storage for visualization
generation_data = []
best_individuals = []
avg_fitness = []
max_fitness = []
operation_log = []  # To log genetic operations

# Create initial population
population = toolbox.populationCreator(n=POPULATION_SIZE)
fitnessValues = list(map(toolbox.evaluate, population))
for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

# Set up the visualization
def init_visualization():
    for ax in axes.flat:
        ax.clear()
    
    # Configure the first subplot (population visualization)
    ax1 = axes[0, 0]
    ax1.set_xlim(0, ONE_MAX_LENGTH)
    ax1.set_ylim(0, POPULATION_SIZE)
    ax1.set_xlabel('Gene Position', fontweight='bold')
    ax1.set_ylabel('Individual in Population', fontweight='bold')
    ax1.set_title('Population Genome (0=blue, 1=red)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Configure the second subplot (fitness progression)
    ax2 = axes[0, 1]
    ax2.set_xlim(0, MAX_GENERATIONS)
    ax2.set_ylim(0, ONE_MAX_LENGTH)
    ax2.set_xlabel('Generation', fontweight='bold')
    ax2.set_ylabel('Fitness', fontweight='bold')
    ax2.set_title('Fitness Progression', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Configure the third subplot (operation visualization)
    ax3 = axes[1, 0]
    ax3.set_xlim(0, ONE_MAX_LENGTH)
    ax3.set_ylim(0, 6)
    ax3.set_xlabel('Gene Position', fontweight='bold')
    ax3.set_title('Genetic Operations (Selection, Crossover, Mutation)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yticks([1, 2, 3, 4, 5])
    ax3.set_yticklabels(['', 'Parent 1', 'Parent 2', 'Child 1', 'Child 2'])
    
    # Configure the fourth subplot (operation log)
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_axis_off()
    ax4.set_title('Operation Log', fontweight='bold')
    
    # Add legend
    blue_patch = mpatches.Patch(color='blue', label='0 (Off)')
    red_patch = mpatches.Patch(color='red', label='1 (On)')
    ax1.legend(handles=[blue_patch, red_patch], loc='upper right')
    
    return fig,

# Update function for animation
def update(generation):
    global population, operation_log
    
    for ax in axes.flat:
        ax.clear()
    
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    
    # Plot the current population
    for i, individual in enumerate(population):
        for j, gene in enumerate(individual):
            color = 'red' if gene == 1 else 'blue'
            ax1.scatter(j, i, color=color, s=30, alpha=0.7)
    
    # Calculate and store fitness statistics
    fitness_values = [ind.fitness.values[0] for ind in population]
    current_avg = sum(fitness_values) / len(population)
    current_max = max(fitness_values)
    avg_fitness.append(current_avg)
    max_fitness.append(current_max)
    
    # Store the best individual
    best_index = fitness_values.index(current_max)
    best_individuals.append(population[best_index])
    
    # Plot fitness progression
    generations = range(len(avg_fitness))
    ax2.plot(generations, avg_fitness, 'o-', color='cyan', label='Average Fitness', linewidth=2, markersize=4)
    ax2.plot(generations, max_fitness, 'o-', color='yellow', label='Max Fitness', linewidth=2, markersize=4)
    
    # Configure the first subplot
    ax1.set_xlim(0, ONE_MAX_LENGTH)
    ax1.set_ylim(0, POPULATION_SIZE)
    ax1.set_xlabel('Gene Position', fontweight='bold')
    ax1.set_ylabel('Individual in Population', fontweight='bold')
    ax1.set_title(f'Generation {generation}: Population Genome', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Configure the second subplot
    ax2.set_xlim(0, MAX_GENERATIONS)
    ax2.set_ylim(0, ONE_MAX_LENGTH)
    ax2.set_xlabel('Generation', fontweight='bold')
    ax2.set_ylabel('Fitness', fontweight='bold')
    ax2.set_title('Fitness Progression', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    # Add text annotations to fitness plot
    ax2.text(0.02, 0.98, f'Max Fitness: {current_max}/{ONE_MAX_LENGTH}', 
             transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax2.text(0.02, 0.88, f'Avg Fitness: {current_avg:.2f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.5))
    
    # Add legend to first subplot
    blue_patch = mpatches.Patch(color='blue', label='0 (Off)')
    red_patch = mpatches.Patch(color='red', label='1 (On)')
    ax1.legend(handles=[blue_patch, red_patch], loc='upper right')
    
    # Stop if we've reached the maximum generations
    if generation >= MAX_GENERATIONS or current_max == ONE_MAX_LENGTH:
        ani.event_source.stop()
        
        # Display final results
        ax4.text(0.05, 0.9, f"Final Generation: {generation}", fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.05, 0.8, f"Best Fitness: {current_max}/{ONE_MAX_LENGTH}", fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.05, 0.7, f"Average Fitness: {current_avg:.2f}", fontweight='bold', transform=ax4.transAxes)
        
        plt.tight_layout()
        return fig,
    
    # Evolutionary algorithm steps with visualization
    # Selection
    operation_log.append(f"Gen {generation}: Selection")
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    
    # Visualize selection by highlighting selected individuals
    selected_indices = [population.index(ind) for ind in offspring[:4]]  # Show first 4 selected
    for idx in selected_indices:
        ax1.axhline(y=idx, color='yellow', alpha=0.3, linewidth=3)
    
    # Crossover visualization
    crossover_point = random.randint(1, ONE_MAX_LENGTH-1)
    ax3.axvline(x=crossover_point, color='green', linestyle='--', alpha=0.7, label='Crossover Point')
    
    # Crossover
    crossover_count = 0
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER and crossover_count < 1:  # Only visualize one crossover
            # Visualize parents before crossover
            parent1, parent2 = toolbox.clone(child1), toolbox.clone(child2)
            
            # Perform crossover
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            
            # Visualize the crossover operation
            for j in range(ONE_MAX_LENGTH):
                # Parent 1
                color = 'red' if parent1[j] == 1 else 'blue'
                ax3.scatter(j, 2, color=color, s=50, alpha=0.8)
                
                # Parent 2
                color = 'red' if parent2[j] == 1 else 'blue'
                ax3.scatter(j, 1, color=color, s=50, alpha=0.8)
                
                # Child 1 (after crossover)
                color = 'red' if child1[j] == 1 else 'blue'
                ax3.scatter(j, 4, color=color, s=50, alpha=0.8, marker='s')
                
                # Child 2 (after crossover)
                color = 'red' if child2[j] == 1 else 'blue'
                ax3.scatter(j, 3, color=color, s=50, alpha=0.8, marker='s')
            
            operation_log.append(f"Gen {generation}: Crossover between individuals")
            crossover_count += 1
    
    # Mutation
    mutation_count = 0
    for mutant in offspring:
        if random.random() < P_MUTATION and mutation_count < 3:  # Only visualize a few mutations
            original = toolbox.clone(mutant)
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
            # Find and visualize the mutated gene
            for j in range(ONE_MAX_LENGTH):
                if original[j] != mutant[j]:
                    # Highlight the mutated position in the population plot
                    individual_idx = offspring.index(mutant)
                    ax1.scatter(j, individual_idx, color='white', s=100, alpha=0.7, marker='*')
                    
                    # Add to operation log
                    operation_log.append(f"Gen {generation}: Mutation at position {j} in individual {individual_idx}")
                    mutation_count += 1
                    break
    
    # Evaluate fitness for new individuals
    freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
    freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
    for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
        individual.fitness.values = fitnessValue

    # Replace population
    population[:] = offspring
    
    # Update operation log display (show last 5 operations)
    ax4.set_axis_off()
    ax4.set_title('Operation Log', fontweight='bold')
    for i, log_entry in enumerate(operation_log[-5:]):
        ax4.text(0.05, 0.8 - i*0.15, log_entry, transform=ax4.transAxes, fontsize=10)
    
    # Configure the operation visualization subplot
    ax3.set_xlim(0, ONE_MAX_LENGTH)
    ax3.set_ylim(0, 6)
    ax3.set_xlabel('Gene Position', fontweight='bold')
    ax3.set_title('Genetic Operations (Selection, Crossover, Mutation)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yticks([1, 2, 3, 4, 5])
    ax3.set_yticklabels(['', 'Parent 1', 'Parent 2', 'Child 1', 'Child 2'])
    
    return fig,

# Create animation
ani = FuncAnimation(fig, update, frames=MAX_GENERATIONS+1,
                    init_func=init_visualization, interval=1000, repeat=False)

plt.tight_layout()
plt.show()