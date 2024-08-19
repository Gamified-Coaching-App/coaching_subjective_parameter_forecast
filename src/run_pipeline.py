from RunningDataset import RunningDataset
from autoencoder import cross_validate, final_training
from config import cross_val_architectures, cross_val_optimiser_params, final_training_architecture, final_training_optimiser_params
import json
from sklearn.model_selection import train_test_split
import itertools
from datetime import datetime
import tensorflow as tf


def run(mode):
    data = RunningDataset()
    X, Y = data.preprocess(days=14)
    report = {}
    timestamp = datetime.now().strftime("%m%d%H%M")
    tf.config.set_visible_devices([], 'GPU')

    if mode == 'cross_validation_fine':
        all_combinations = list(itertools.product(cross_val_architectures, cross_val_optimiser_params))
        for architecture, optimiser_params in all_combinations:
            cross_validate(Y=Y, X=X, architecture=architecture, optimiser_params=optimiser_params, report=report, n_splits=10)
        with open(f'../report/report_files/cross_validation_report_fine_{timestamp}.json', 'w') as json_file:
            json.dump(report, json_file, indent=2)
    elif mode == 'cross_validation_base':
        for i in range(len(cross_val_architectures)):
            cross_validate(Y=Y, X=X, architecture=cross_val_architectures[i], optimiser_params=cross_val_optimiser_params[i], report=report, n_splits=10)
        with open(f'../report/report_files/cross_validation_report_base_{timestamp}.json', 'w') as json_file:
            json.dump(report, json_file, indent=2)

    elif mode == 'final_training':
        # Target split: 15% test, 15% val, 70% train
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

        # Then, split the training+validation set into training and validation sets (75% train, 25% val of the train+val set)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=42)

        report = final_training(X_train, Y_train, X_val, Y_val, X_test, Y_test, final_training_architecture, final_training_optimiser_params, report)

        with open(f'../report/report_files/final_training_report_{timestamp}.json', 'w') as json_file:
            json.dump(report, json_file, indent=2)

    
if __name__ == "__main__":
    #run(mode='final_training')
    run(mode='cross_validation_base')