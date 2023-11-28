import keras
import numpy as np
from tensorflow.keras import layers


class NeuralNet:

    def __init__(self, inp_nodes: int, op_nodes=1, hidden_layers=1):
        """
        initializes the model
        """

        self.inp_nodes = inp_nodes
        self.hidden_nodes = 2 * inp_nodes + 1
        self.out_nodes = op_nodes

        self.model = keras.Sequential()
        self.model.add(layers.Dense(self.inp_nodes, activation='sigmoid'))
        for i in range(hidden_layers):
            self.model.add(layers.Dense(self.hidden_nodes, activation='sigmoid'))
        self.model.add(layers.Dense(self.out_nodes, activation='sigmoid'))
        self.model.build((None, self.inp_nodes))

    def forward(self, x):
        """
        forward propagation step
        """
        x = np.array(x)
        x.reshape(1, -1)

        return self.model(x)

    def get_model_summary(self):
        """
        prints model summary
        """
        print(self.model.summary())

    def train(self, X_train, y_train, epochs, optimizer, X_test=None, y_test=None):
        """
        training the model
        """
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
        )

        self.model.fit(X_train,
                       y_train,
                       epochs=epochs,
                       validation_data=(X_test, y_test))
        # score = self.model.evaluate(X_test, y_test)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

        print("Training completed...")
