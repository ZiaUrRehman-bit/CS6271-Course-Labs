# 1. Define problem
def fitness(individual):
    return ...

# 2. Define GA parameters
POP_SIZE = ...
P_CROSSOVER = ...
P_MUTATION = ...
MAX_GEN = ...

# 3. Setup DEAP toolbox
creator.create("FitnessMax/Min", base.Fitness, weights=(...))
creator.create("Individual", list/array, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", ...)      # how to generate one gene
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register operators
toolbox.register("evaluate", fitness)
toolbox.register("select", ...)
toolbox.register("mate", ...)
toolbox.register("mutate", ...)

# 4. Initialize population
population = toolbox.population(n=POP_SIZE)
for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)

# 5. Evolution loop
for gen in range(MAX_GEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover & mutation
    ...
    # Evaluate new fitness
    ...
    # Replace population
    population[:] = offspring

    # Collect stats
    ...

# 6. Show results
