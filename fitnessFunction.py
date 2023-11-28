import keras.optimizers

from makedata import load_data
from makemodel.MLModel import NeuralNet
from sklearn.metrics import accuracy_score


def fitness_function(params, name="hillValley"):
    # Get processed data
    X_train, X_test, y_train, y_test = load_data(name)
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    epochs, learning_rate, hidden_layers = params
    hidden_layers = int(hidden_layers)

    # Get the model
    model = NeuralNet(inp_nodes=X_train.shape[1], hidden_layers=hidden_layers)
    model.get_model_summary()

    epochs = int(epochs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.train(X_train, y_train, epochs, optimizer, X_test, y_test)
    y_pred = model.forward(X_test)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

    return accuracy_score(y_test, y_pred)
