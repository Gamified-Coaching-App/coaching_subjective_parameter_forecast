cross_val_architectures = [
    [
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True, 'input_shape': (14, 10)},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': False},
        {'type': 'repeat_vector', 'n': 14},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True},
        {'type': 'time_distributed'}
    ],
    [
        {'type': 'bidirectional_lstm', 'units': 19, 'return_sequences': True, 'input_shape': (14, 10)},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': False},
        {'type': 'repeat_vector', 'n': 14},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True},
        {'type': 'time_distributed'}
    ]
]


cross_val_optimiser_params = [
    {
        'optimiser': 'adadelta',
        'learning_rate': 1.0,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64,
        'epochs': 2
    },
    {
        'optimiser': 'adadelta',
        'learning_rate': 1.0,
        'weight_decay': 0.01,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64,
        'epochs': 2
    }
]

# dict for schedule {
#             'min_lr': 0.0001,
#             'warmup_steps': 1000,
#             'decay_steps': 100000,
#             'max_lr': 1.0
#         },

final_training_architecture= [
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True, 'input_shape': (14, 10)},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': False},
        {'type': 'repeat_vector', 'n': 14},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True},
        {'type': 'bidirectional_lstm', 'units': 20, 'return_sequences': True},
        {'type': 'time_distributed'}
    ]

final_training_optimiser_params={
        'optimiser': 'adadelta',
        'learning_rate': 1.0,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64,
        'epochs': 1
    }