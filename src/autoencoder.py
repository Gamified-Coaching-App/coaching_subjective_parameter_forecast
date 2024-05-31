import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers, Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from sklearn.metrics import roc_auc_score
import numpy as np

class Autoencoder(models.Model):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        self.model = Sequential([
            layers.Bidirectional(LSTM(20, return_sequences=True), input_shape=(config.get(config.get('days')), 10)),
            layers.Bidirectional(LSTM(20, return_sequences=False)),
            layers.RepeatVector(config.get('days')),
            layers.Bidirectional(LSTM(20, return_sequences=True)),
            layers.Bidirectional(LSTM(20, return_sequences=True)),
            layers.TimeDistributed(Dense(10))
        ])
        #self.decoder = Sequential()
        # for i in range(num_layers):
        #     return_sequences = True if i < num_layers - 1 else False
        #     self.encoder.add(LSTM(current_units, recurrent_dropout=0.3, dropout=0.3, return_sequences=return_sequences, input_shape=(days, 10) if i == 0 else None))
        #     if unit_decline != 0: current_units = max(int(current_units * (1 - unit_decline)), 1)  # Ensuring units don't fall below 1

        # self.bridge = RepeatVector(21)  # Assuming the number of time steps to be repeated is 14

        # for i in range(num_layers):
        #     if unit_decline != 0: current_units = min(int(current_units / (1 - unit_decline)), units_first_layer)
        #     return_sequences = True if i < num_layers - 1 else True  # All decoder layers should return sequences except maybe the last
        #     self.decoder.add(LSTM(current_units, recurrent_dropout=0.3, dropout=0.3, return_sequences=return_sequences))

        # self.decoder.add(TimeDistributed(Dense(10)))  # Assuming the original feature size is 10

    def call(self, inputs):
        x = self.model(inputs)
        return x
    
def train_autoencoder(autoencoder, train_set, config):
    num_epochs = config.get('num_epochs')
    batch_size = config.get('batch_size')
    optimizer_name = config.get('optimizer')  # Default to Adadelta if not specified
    learning_rate = config.get('learning_rate')
    optimizer_params = config.get('optimizer_params')

    if optimizer_name.lower() == 'adadelta':
        optimizer = optimizers.Adadelta(learning_rate=learning_rate, **(optimizer_params or {}))
    elif optimizer_name.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate, **(optimizer_params or {}))
    elif optimizer_name.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate, **(optimizer_params or {}))
    else:
        raise ValueError("Unsupported optimizer type provided!")

    autoencoder.compile(optimizer=optimizer, loss='mse')

    history = autoencoder.fit(train_set, epochs=num_epochs, verbose=1)
    
    return history

def evaluate(autoencoder, X_masked, X):
    return autoencoder.evaluate(X_masked, X, verbose=0)

def run_and_evaluate(X_train, X_train_masked, X_test, X_test_masked, config):
    autoencoder = Autoencoder(config)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_masked, X_train)).batch(500)
    history = train_autoencoder(autoencoder, train_dataset, config)
    
    print(f'Test loss: {evaluate(autoencoder, X_test_masked, X_test)}')
    print(f'Train loss: {evaluate(autoencoder, X_train_masked, X_train)}')
    autoencoder.model.export('model/subjective_parameter')