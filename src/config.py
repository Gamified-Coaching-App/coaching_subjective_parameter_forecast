cross_val_architectures = [
    # [
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'flatten'},
    #     {'type': 'dense', 'units': 21},
    #     {'type': 'reshape', 'target_shape': (7, 3)},
    #     {'type':'activation', 'activation':'linear'}
    # ],
    [
        {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'flatten'},
        {'type': 'dense', 'units': 21},
        {'type': 'reshape', 'target_shape': (7, 3)},
        {'type':'activation', 'activation':'linear'}
    ],
    # [
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'flatten'},
    #     {'type': 'dense', 'units': 21},
    #     {'type': 'reshape', 'target_shape': (7, 3)},
    #     {'type':'activation', 'activation':'linear'}
    # ],
    # [
    #     {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 24, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 24, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 24, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 24, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 24, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 24, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'flatten'},
    #     {'type': 'dense', 'units': 21},
    #     {'type': 'reshape', 'target_shape': (7, 3)},
    #     {'type':'activation', 'activation':'linear'}
    # ],
    # [
    #     {'type': 'lstm', 'units': 128, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 128, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 128, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 128, 'return_sequences': False},
    #     {'type': 'dense', 'units': 21},
    #     {'type': 'reshape', 'target_shape': (7, 3)},
    #     {'type':'activation', 'activation':'linear'}
    # ],
    # [
    #     {'type': 'lstm', 'units': 64, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 64, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 64, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 64, 'return_sequences': True},
    #     {'type': 'lstm', 'units': 21, 'return_sequences': False},
    #     {'type': 'reshape', 'target_shape': (7, 3)},
    #     {'type':'activation', 'activation':'linear'}
    # ]
]


cross_val_optimiser_params = [
    # Without scheduler
    {
        'optimiser': 'adam',
        'learning_rate': 0.0001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.000001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.000001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.000001,
        'weight_decay': 0.0000001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.000001,
        'weight_decay': 0.000001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.00001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.00001,
        'weight_decay': 0.0000001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.00001,
        'weight_decay': 0.000001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.00001,
        'weight_decay': 0.00001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.00001,
        'weight_decay': 0.001,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.0001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None',
        'batch_size': 64
    },

    # With scheduler (cosine_decay)
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.00001,
            'warmup_steps': 1500,
            'max_lr': 0.001,
            'decay_steps': 100000,
            'alpha': 0.00001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.00001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.00001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.00001,
            'warmup_steps': 1500,
            'max_lr': 0.1,
            'decay_steps': 100000,
            'alpha': 0.00001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.0001,
            'warmup_steps': 1500,
            'max_lr': 0.001,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.0001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.0001,
            'warmup_steps': 1500,
            'max_lr': 0.1,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    }
]
    # {
    #     'optimiser': 'adam',
    #     'learning_rate': 0.0001,
    #     'weight_decay': 0.0,
    #     'scheduler': 'None',
    #     'scheduler_params': 'None', 
    #     'batch_size': 64
    # },
    # {
    #     'optimiser': 'adadelta',
    #     'learning_rate': 1.0,
    #     'weight_decay': 0.0,
    #     'scheduler': 'None',
    #     'scheduler_params': 'None', 
    #     'batch_size': 64
    # },
    # {
    #     'optimiser': 'adadelta',
    #     'learning_rate': 1.0,
    #     'weight_decay': 0.0,
    #     'scheduler': 'None',
    #     'scheduler_params': 'None', 
    #     'batch_size': 64
    # }
# ]

final_training_architecture= [
        {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 12, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'flatten'},
        {'type': 'dense', 'units': 21},
        {'type': 'reshape', 'target_shape': (7, 3)},
        {'type':'activation', 'activation':'linear'}
    ]

final_training_optimiser_params={
        'optimiser': 'adam',
        'learning_rate': 0.1,  # Ignored when scheduler is applied
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.0001,
            'warmup_steps': 1500,
            'max_lr': 0.001,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    }