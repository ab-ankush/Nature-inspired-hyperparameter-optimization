import keras.optimizers

from makedata.getBreastCancerData import load_data
from makemodel.MLModel import NeuralNet
from sklearn.metrics import accuracy_score
from mealpy.swarm_based import AO


def get_optimizer(opt_id, learning_rate):
    if opt_id == 0:
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt_id == 1:
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_id == 2:
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_id == 3:
        return keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
        return keras.optimizers.Adagrad(learning_rate=learning_rate)


def fitness_function(params):
    # Get processed data
    X_train, X_test, y_train, y_test = load_data()
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # Get the model
    model = NeuralNet(inp_nodes=X_train.shape[1])
    model.get_model_summary()

    epochs, learning_rate, opt_id = params
    epochs = int(epochs)
    opt_id = int(opt_id)
    optimizer = get_optimizer(opt_id, learning_rate)

    model.train(X_train, y_train, epochs, optimizer, X_test, y_test)
    y_pred = model.forward(X_test)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

    return accuracy_score(y_test, y_pred)


# Select/Find best #epochs, learning_rate and optimizer

# bounds for hyper-params
lb = [1, 0.001, 0]
ub = [100, 0.1, 4]

obj_func = fitness_function
verbose = True
epoch = 10000  # for the algo
pop_size = 50

problem_dict = {
    "fit_func": fitness_function,
    "lb": lb,
    "ub": ub,
    "minmax": "max"
}

model = AO.OriginalAO(epoch, pop_size)
best_position, best_fitness = model.solve(problem_dict)
print(f"Solution: {best_position}, Fitness: {best_fitness}")
