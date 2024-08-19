import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers, Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

class Autoencoder(models.Model):
    def __init__(self, architecture, optimiser_params):
        super(Autoencoder, self).__init__()
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

            elif layer_type == 'time_distributed':
                x = layers.TimeDistributed(Dense(10))(x)
                model = models.Model(inputs=original_input, outputs=x)
                return model

        # Default output layer if not explicitly specified
        x = layers.Dense(config['output_dim'], activation=config.get('output_activation', 'linear'))(x)
        model = models.Model(inputs=original_input, outputs=x)
        return model

    def call(self, inputs):
        return self.model(inputs)

def train_autoencoder(autoencoder, train_set, val_set, optimiser_params, mode):
    autoencoder.compile(optimizer=autoencoder.optimizer, loss='mse')
    num_epochs = optimiser_params['epochs']

    if mode == 'final_training':
        checkpoint_filepath = 'best_weights.weights.h5'
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )
        history = autoencoder.fit(
                train_set,               
                epochs=num_epochs,       
                validation_data=val_set,
                verbose=1,
                callbacks=[checkpoint_callback]            
            )
        
    
    else: 
        history = autoencoder.fit(
                train_set,               
                epochs=num_epochs,       
                validation_data=val_set,
                verbose=1      
            )
    
    return history


def evaluate(autoencoder, X_masked, X):
    return autoencoder.evaluate(X_masked, X, verbose=0)

def train(X_train, Y_train, X_val, Y_val, architecture, optimiser_params, mode):
    autoencoder = Autoencoder(architecture=architecture, optimiser_params=optimiser_params)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(optimiser_params['batch_size'])
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(optimiser_params['batch_size'])
    
    history = train_autoencoder(autoencoder=autoencoder, train_set=train_dataset, val_set=val_dataset, optimiser_params=optimiser_params, mode=mode)
    
    min_val_loss = min(history.history['val_loss'])
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    return autoencoder, min_val_loss, train_losses, val_losses

def cross_validate(Y, X, architecture, optimiser_params, report, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_losses = []

    for train_index, val_index in kf.split(Y):
        Y_train, Y_val = Y[train_index], Y[val_index]
        X_train, X_val = X[train_index], X[val_index]

        # Train the model and get the minimum validation loss for this fold
        _, min_val_loss, _, _ = train(
            X_train=tf.convert_to_tensor(X_train), 
            Y_train=tf.convert_to_tensor(Y_train), 
            X_val=tf.convert_to_tensor(X_val), 
            Y_val=tf.convert_to_tensor(Y_val), 
            architecture=architecture,
            optimiser_params=optimiser_params,
            mode='cross_validation'
        )
        
        val_losses.append(min_val_loss)
    
    # Compute the average validation loss across all folds
    average_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    key = f'architecture={str(architecture)}, optimiser_params={str(optimiser_params)}'

    report[key] = {
    'val_losses': [round(loss, 8) for loss in val_losses],
    'mean_val_loss': round(average_val_loss, 8),
    'stdv_val_loss': round(std_val_loss, 8)
    }       
    return report

def final_training(Y_train, X_train, Y_val, X_val, Y_test, X_test, architecture, optimiser_params, report):
    autoencoder, min_val_loss, train_losses, val_losses = train(
            X_train=tf.convert_to_tensor(X_train), 
            Y_train=tf.convert_to_tensor(Y_train), 
            X_val=tf.convert_to_tensor(X_val), 
            Y_val=tf.convert_to_tensor(Y_val), 
            architecture=architecture,
            optimiser_params=optimiser_params,
            mode='final_training'
            )
    # Load the best weights saved during training
    autoencoder.load_weights('best_weights.weights.h5')
    autoencoder.model.export('../model/subjective_parameter_forecaster')

    test_loss = evaluate(autoencoder, X_test, Y_test)
    print(f'Test loss: {test_loss}')

    report['train_losses'] = [round(loss, 8) for loss in train_losses]
    report['val_losses'] = [round(loss, 8) for loss in val_losses]
    report['min_val_loss'] = round(min_val_loss, 8)
    report['test_loss'] = round(test_loss, 8)

    return report