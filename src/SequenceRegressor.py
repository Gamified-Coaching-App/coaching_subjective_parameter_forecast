import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers, Sequential, Layer
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
import json

NUM_EPOCHS = 1000

"""
BatchLossHistory class serves as a callback to store the loss of each batch and epoch during training.
"""
class BatchLossHistory(Callback):
    """
    function to initialize the batch_losses and epoch_losses lists
    """
    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.epoch_losses = []

    """
    function to store the loss of each batch
    """
    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

    """
    function to store the loss of each epoch
    """
    def on_epoch_end(self, epoch, logs=None):
        mean_loss = sum(self.batch_losses) / len(self.batch_losses)
        self.epoch_losses.append(mean_loss)
        self.batch_losses = []

"""
SequenceRegressor class serves as a wrapper for the model and the optimiser.
"""
class SequenceRegressor(models.Model):
    """
    __init__ sets up model accoring to @architecture and @optimiser_params
    """ 
    def __init__(self, architecture, optimiser_params):
        super(SequenceRegressor, self).__init__()
        self.model = self.create_model(architecture)
    
        if optimiser_params['scheduler'] == 'cosine_decay':
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=optimiser_params['scheduler_params']['min_lr'],
                decay_steps=optimiser_params['scheduler_params']['decay_steps'],
                warmup_target=optimiser_params['scheduler_params']['max_lr'],
                warmup_steps=optimiser_params['scheduler_params']['warmup_steps'],
                alpha=optimiser_params['scheduler_params']['min_lr'],
            )
        else:
            lr_schedule = optimiser_params['learning_rate']

        if optimiser_params['optimiser'] == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=lr_schedule, weight_decay=optimiser_params['weight_decay'])
        elif optimiser_params['optimiser'] == 'adamw':
            self.optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=optimiser_params['weight_decay'])
        else:
            self.optimizer = optimizers.Adadelta(learning_rate=lr_schedule, weight_decay=optimiser_params['weight_decay'])

    """
    function to create the model according to the configuration of layers
    """ 
    def create_model(self, config):
        original_input = layers.Input(shape=(14, 10), name='original_input')
        x = original_input

        for layer_config in config:
            layer_type = layer_config['type']

            if layer_type == 'lstm':
                params = {
                    'units': layer_config['units'],
                    'return_sequences': layer_config['return_sequences'],
                }
                if 'dropout' in layer_config:
                    params['dropout'] = layer_config['dropout']
                if 'recurrent_dropout' in layer_config:
                    params['recurrent_dropout'] = layer_config['recurrent_dropout']
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                x = layers.LSTM(**params)(x)

            elif layer_type == 'bidirectional_lstm':
                params = {
                    'units': layer_config['units'],
                    'return_sequences': layer_config['return_sequences'],
                }
                if 'dropout' in layer_config:
                    params['dropout'] = layer_config['dropout']
                if 'recurrent_dropout' in layer_config:
                    params['recurrent_dropout'] = layer_config['recurrent_dropout']
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                x = layers.Bidirectional(layers.LSTM(**params))(x)

            elif layer_type == 'dense':
                params = {
                    'units': layer_config['units'],
                }
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                if 'kernel_initializer' in layer_config:
                    params['kernel_initializer'] = layer_config['kernel_initializer']
                if 'bias_initializer' in layer_config:
                    params['bias_initializer'] = layer_config['bias_initializer']
                
                x = layers.Dense(**params)(x)

            elif layer_type == 'dropout':
                x = layers.Dropout(rate=layer_config['rate'])(x)

            elif layer_type == 'reshape':
                x = layers.Reshape(target_shape=layer_config['target_shape'])(x)

            elif layer_type == 'alphadropout':
                params = {
                    'rate': layer_config['rate'],
                }
                x = layers.AlphaDropout(**params)(x)
            
            elif layer_type == 'repeat_vector':
                params = {
                    'n': layer_config['n'],
                }
                x = layers.RepeatVector(**params)(x)

            elif layer_type == 'transformer_encoder':
                params = {
                    'num_heads': layer_config['num_heads'],
                    'intermediate_dim': layer_config['intermediate_dim'],
                    'dropout': layer_config.get('dropout'),
                    'activation': layer_config.get('activation'), 
                    'normalize_first': layer_config.get('normalize_first')
                }
                x = TransformerEncoder(**params)(x)
            
            elif layer_type == 'transformer_decoder':
                params = {
                    'num_heads': layer_config['num_heads'],
                    'intermediate_dim': layer_config['intermediate_dim'],
                    'dropout': layer_config.get('dropout'),
                    'activation': layer_config.get('activation'), 
                    'normalize_first': layer_config.get('normalize_first')
                }
                x = TransformerDecoder(**params)(x)

            elif layer_type == 'flatten':
                x = layers.Flatten()(x)

            elif layer_type == 'activation':
                x = layers.Activation(layer_config['activation'])(x)
                model = models.Model(inputs=original_input, outputs=x)
                return model

            elif layer_type == 'time_distributed':
                x = layers.TimeDistributed(Dense(10))(x)
                model = models.Model(inputs=original_input, outputs=x)
                return model

    """
    function called when the model is called/trained to pass the input through the model
    """ 
    def call(self, inputs):
        return self.model(inputs)

"""
function to train model, use early stopping and save the best weights, differentiate between final training and cross validation
"""
def train_sequence_regressor(sequence_regressor, train_set, val_set, optimiser_params, mode):
    sequence_regressor.compile(optimizer=sequence_regressor.optimizer, loss='mse')
    num_epochs = NUM_EPOCHS

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=300,  
        restore_best_weights=False,
        verbose=0,
        min_delta=0.0000001
    )

    batch_loss_history = BatchLossHistory()

    if mode == 'final_training':
        checkpoint_filepath = 'best_weights.weights.h5'
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=0
        )
        history = sequence_regressor.fit(
                train_set,               
                epochs=num_epochs,       
                validation_data=val_set,
                verbose=0,
                callbacks=[checkpoint_callback, batch_loss_history, early_stopping]            
            )
        epoch_losses = batch_loss_history.epoch_losses
        batch_losses = batch_loss_history.batch_losses
    
    else: 
        history = sequence_regressor.fit(
                train_set,               
                epochs=num_epochs,       
                validation_data=val_set,
                verbose=0,
                callbacks=[early_stopping, batch_loss_history]       
            )
    
    return history, epoch_losses, batch_losses

"""
function to evaluate model on test set
"""
def evaluate(sequence_regressor, X_masked, X):
    return sequence_regressor.evaluate(X_masked, X, verbose=0)

"""
function to initialise model, cut data is batches, train model
"""
def train(X_train, Y_train, X_val, Y_val, architecture, optimiser_params, mode):
    sequence_regressor = SequenceRegressor(architecture=architecture, optimiser_params=optimiser_params)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(optimiser_params['batch_size'])
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(optimiser_params['batch_size'])
    
    history, epoch_losses, batch_losses = train_sequence_regressor(sequence_regressor=sequence_regressor, train_set=train_dataset, val_set=val_dataset, optimiser_params=optimiser_params, mode=mode)
    
    min_val_loss = min(history.history['val_loss'])
    train_losses_epoch = epoch_losses
    train_losses_batch = batch_losses
    val_losses = history.history['val_loss']

    num_epochs = len(train_losses_epoch)

    return sequence_regressor, min_val_loss, val_losses, train_losses_epoch, train_losses_batch, num_epochs

"""
function for 5-fold cross validation. Train, add results to report and return report
"""
def cross_validate(Y, X, architecture, optimiser_params, report, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_losses = []

    for train_index, val_index in kf.split(Y):
        Y_train, Y_val = Y[train_index], Y[val_index]
        X_train, X_val = X[train_index], X[val_index]

        _, min_val_loss, _, _, _, num_epochs = train(
            X_train=tf.convert_to_tensor(X_train), 
            Y_train=tf.convert_to_tensor(Y_train), 
            X_val=tf.convert_to_tensor(X_val), 
            Y_val=tf.convert_to_tensor(Y_val), 
            architecture=architecture,
            optimiser_params=optimiser_params,
            mode='cross_validation'
        )
        print('finished training at epoch:', num_epochs, 'with min_val_loss:', min_val_loss)

        val_losses.append(min_val_loss)
        
    average_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    key = f'architecture={str(architecture)}, optimiser_params={str(optimiser_params)}'

    report[key] = {
    'val_losses': val_losses,
    'mean_val_loss': round(average_val_loss, 6),
    'stdv_val_loss': round(std_val_loss, 6),
    'num_epochs': num_epochs
    }       
    return report


"""
function for final training. Train, evaluate on test set, save model, save results to report and return report
"""
def final_training(Y_train, X_train, Y_val, X_val, Y_test, X_test, architecture, optimiser_params, report):
    sequence_regressor, min_val_loss, val_losses, train_losses_epoch, train_losses_batch, num_epochs = train(
            X_train=tf.convert_to_tensor(X_train), 
            Y_train=tf.convert_to_tensor(Y_train), 
            X_val=tf.convert_to_tensor(X_val), 
            Y_val=tf.convert_to_tensor(Y_val), 
            architecture=architecture,
            optimiser_params=optimiser_params,
            mode='final_training'
            )
    
    sequence_regressor.load_weights('best_weights.weights.h5')
    sequence_regressor.model.export('../model/subjective_parameter_forecaster')

    test_loss = evaluate(sequence_regressor, X_test, Y_test)
    print(f'Test loss: {test_loss}')

    predictions = sequence_regressor.predict(X_test)

    predictions_list = predictions.tolist()
    ground_truth_list = Y_test.tolist()

    results = {
        'predictions': predictions_list,
        'ground_truth': ground_truth_list
    }
    with open('../report/report_files/inference_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    report['train_losses_epoch'] = train_losses_epoch
    report['train_losses_batch'] = train_losses_batch
    report['val_losses'] = val_losses
    report['min_val_loss'] = round(min_val_loss, 6)
    report['test_loss'] = round(test_loss, 6)
    report['num_epochs'] = num_epochs

    return report