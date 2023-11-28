from fitnessFunction import fitness_function
from mealpy.human_based import CHIO

# Select/Find best #epochs, learning_rate and hidden layers

# bounds for hyper-params
lb = [1, 0.001, 1]
ub = [100, 0.1, 5]

obj_func = fitness_function
verbose = True
epoch = 10000  # for the algo
pop_size = 50
brr = 0.15

problem_dict = {
    "fit_func": fitness_function,
    "lb": lb,
    "ub": ub,
    "minmax": "max"
}

model = CHIO.BaseCHIO(epoch, pop_size, brr)
best_position, best_fitness = model.solve(problem_dict)
print(f"Solution: {best_position}, Fitness: {best_fitness}")
