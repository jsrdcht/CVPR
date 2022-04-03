cfg = {
    'model': 'tf_efficientnet_b3_ns',
    'epochs': 30,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 30
    },
    'batch_size': 8,

    'attack': True,
    'attack_loss_ratio': 1,
    'attack_algos': ['FGSM','PGD'],
    'attack_range': [0.1, 0.3]
}

attack_param = {

}
