from RunningDataset import RunningDataset
from autoencoder import run_and_evaluate

config = {
    # Data parameters
    'days': 14, 

    # Model architecture
    'units_first_layer': 500,  # Number of units in the first layer of the encoder
    'number_of_layers': 2,           # Number of layers in the encoder and decoder
    'unit_decline': 0,      # Share per layer to reduce the number of units by

    # Training parameters
    'num_epochs': 100,         # Number of epochs for training
    'batch_size': 512,        # Batch size for training
    'learning_rate': 0.01,   # Learning rate for the optimizer
    'optimizer': 'adadelta',      # Type of optimizer (e.g., 'adam', 'adadelta', 'sgd')
    'optimizer_params': {}#{'beta_1':0.6, 'beta_2':0.9999, 'epsilon':1e-07}    # Additional parameters for the optimizer
}

def run():
    data = RunningDataset()
    X_train, X_train_masked, X_test, X_test_masked = data.preprocess(days=config.get('days'))
    run_and_evaluate(X_train, X_train_masked, X_test, X_test_masked, config)

if __name__ == "__main__":
    run()
