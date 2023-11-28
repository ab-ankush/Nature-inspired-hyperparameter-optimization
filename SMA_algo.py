from fitnessFunction import fitness_function
from mealpy.bio_based import SMA
# Select/Find best #epochs, learning_rate and hidden_layers

# bounds for hyper-params
# epochs, learning rate and hidden layers
lb = [1, 0.001, 1]
ub = [100, 0.1, 5]

obj_func = fitness_function
verbose = True
epoch = 10000  # for the algo
pop_size = 50
p_t = 0.03

problem_dict = {
    "fit_func": fitness_function,
    "lb": lb,
    "ub": ub,
    "minmax": "max"
}

model = SMA.BaseSMA(epoch, pop_size, p_t)
best_position, best_fitness = model.solve(problem_dict)
print(f"Solution: {best_position}, Fitness: {best_fitness}")
