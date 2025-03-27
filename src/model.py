#!/usr/bin/env python

import argparse
import pandas as pd
import joblib
from pathlib import Path
import logging
import traceback
import os
from preprocessing import preprocess_data
from models import train_random_forest, train_catboost
from clearml import Task


class Logger:
    """
    Singleton logger class to manage logging throughout the application.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.logger = logging.getLogger('model_logger')
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            cls._instance.logger.addHandler(handler)
            cls._instance.logger.setLevel(logging.INFO)
        return cls._instance


class My_Classifier_Model:
    def __init__(self, mode='train', model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.model_name = f'{model_type}_model.joblib'
        self.logger = Logger().logger
        
        self.task = None
        self.setup_clearml(mode)

    def setup_clearml(self, mode):
        """
        Initializes the ClearML task and ensures no task conflicts.
        """
        # It's crucial to export these keys into environment variables!!
        self.access_key = os.getenv('CLEARML_ACCESS_KEY')
        self.secret_key = os.getenv('CLEARML_SECRET_KEY')

        if not self.access_key or not self.secret_key:
            print("Error: Couldn't find ClearML Access and/or Secret Key.")
            return
        
        # Create or reuse task
        if self.task is not None:
            self.task.close()

        self.task = Task.init(
            project_name='model_project',
            task_name=mode,
            task_type=Task.TaskTypes.optimizer
        )

    def log_datasets_to_clearml(self, X_train, X_test, y_train, y_test):
        """
        Logs the final Train/Test datasets to ClearML as artifacts.
        """
        try:
            self.logger.info('Logging Train/Test datasets to ClearML...')
            train_dataset = pd.concat([X_train, y_train], axis=1)
            test_dataset = pd.concat([X_test, y_test], axis=1)

            # Log both datasets as artifacts
            self.task.upload_artifact(name='train_dataset', artifact_object=train_dataset)
            self.task.upload_artifact(name='test_dataset', artifact_object=test_dataset)

            self.logger.info('Train/Test datasets successfully logged to ClearML.')

        except Exception as e:
            self.logger.error(f'Error while logging datasets to ClearML: {e}')
            print(f'Error while logging datasets to ClearML: {e}')

    def log_model_to_clearml(self, model_path):
        """
        Logs the trained model to ClearML as an artifact.
        """
        try:
            self.logger.info(f'Logging model {self.model_name} to ClearML...')
            self.task.upload_artifact(name='best_model', artifact_object=model_path)
            self.logger.info('Model successfully logged to ClearML.')

        except Exception as e:
            self.logger.error(f'Error while logging model to ClearML: {e}')
            print(f'Error while logging model to ClearML: {e}')

    def train(self, dataset_filename):
        self.logger.info('Loading and processing data...')
        print('Loading and processing data...')

        try:
            X_train, X_test, y_train, y_test = preprocess_data(dataset_filename, mode='train')
            self.logger.info('Data loaded and processed successfully.')
        except Exception as e:
            self.logger.error(f'Error while processing data: {e}')
            print(f'Error while processing data: {e}')
            return

        # Log Train/Test datasets to ClearML
        self.log_datasets_to_clearml(X_train, X_test, y_train, y_test)

        if self.model_type == 'random_forest':
            self.logger.info('----- Fitting Random Forest model with GridSearchCV... -----')
            print('\n----- Fitting Random Forest model with GridSearchCV... -----\n')

            try:
                self.model, accuracy = train_random_forest(X_train, X_test, y_train, y_test)
                self.logger.info(f'The best Random Forest model: {self.model}')
                self.logger.info(f'Best params for the model: {self.model.get_params()}')
                self.logger.info(f'Accuracy on X_test data set: {accuracy:.4f}')
                print(f'\n-----\nThe best Random Forest model: {self.model}\n-----\n')
                print(f'\n-----\nBest params for the model:\n-----\n')
                print(self.model.get_params())
                print(f'\n-----\nAccuracy on X_test data set: {accuracy:.4f}\n-----\n')
            except Exception as e:
                self.logger.error(f'Error while training Random Forest: {e}')
                print(f'Error while training Random Forest: {e}')
                return

        elif self.model_type == 'catboost':
            self.logger.info('----- Fitting CatBoost model with GridSearchCV... -----')
            print('\n----- Fitting CatBoost model with GridSearchCV... -----\n')

            try:
                self.model, accuracy = train_catboost(X_train, X_test, y_train, y_test)
                self.logger.info(f'The best CatBoost model: {self.model}')
                self.logger.info(f'Best params for the model: {self.model.get_params()}')
                self.logger.info(f'Accuracy on X_test data set: {accuracy:.4f}')
                print(f'\n-----\nThe best CatBoost model: {self.model}\n-----\n')
                print(f'\n-----\nBest params for the model:\n-----\n')
                print(self.model.get_params())
                print(f'\n-----\nAccuracy on X_test data set: {accuracy:.4f}\n-----\n')
            
            except Exception as e:
                self.logger.error(f'Error while training CatBoost: {e}')
                print(f'Error while training CatBoost: {e}')
                return

        self.save_model()

    def save_model(self):
        model_path = Path('./model') / self.model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(self.model, model_path)
            self.logger.info(f'Model {self.model_name} saved successfully to {model_path}')
            print(f'\n-----\nModel {self.model_name} saved successfully!\n-----\n')

            # Log the trained model to ClearML
            if self.task:
                self.log_model_to_clearml(model_path)

        except Exception as e:
            self.logger.error(f'Error saving model {self.model_name}: {e}')
            print(f'Error saving model {self.model_name}: {e}')
        finally:
            if self.task:
                self.logger.info("Closing ClearML task...")
                self.task.close()

    def predict(self, dataset_filename):
        model_path = Path('./model') / self.model_name
        
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f'Model loaded successfully from {model_path}')
            print(f'\n-----\nModel loaded successfully from {model_path}\n-----\n')
        
        except Exception as e:
            self.logger.error(f'Error loading model from {model_path}: {e}')
            print(f'Error loading model from {model_path}: {e}')
            return

        self.logger.info('Loading and processing data for prediction...')
        print('\n-----\nLoading and processing data for prediction...\n-----\n')

        try:
            X_final, test_data = preprocess_data(dataset_filename, mode='test')

            self.logger.info('Data loaded and processed successfully for prediction.')
        
        except Exception as e:
            error_message = f"Error while processing data: {e}\n{traceback.format_exc()}"
            self.logger.error(error_message)
            print(error_message)
            return

        self.logger.info(f'Making predictions using {self.model_type} model...')
        print(f'\n-----\nMaking predictions using {self.model_type} model...\n-----\n')

        predicted_file_name = f'prediction_{self.model_type}.csv'
        self.logger.info(f'{self.model_type.capitalize()}: Writing the predicted data (DataFrame) into .csv file...')
        print(f'\n-----\n{self.model_type.capitalize()}: Writing the predicted data (DataFrame) into .csv file...\n-----\n')
        
        try:
            y_pred_all = self.model.predict(X_final)
            final_prediction_df = pd.DataFrame(data={'PassengerId': test_data.PassengerId, 'Transported': y_pred_all})
            final_prediction_df.to_csv(f'./results_data/{predicted_file_name}', index=False)
            
            self.logger.info('Written successfully!')
            print('\n-----\nWritten successfully!\n-----\n')

        except Exception as e:
            self.logger.error(f'Error occurred while writing the data for {self.model_type}: {e}')
            print(f'Error occurred while writing the data for {self.model_type}.')
        finally:
            if self.task:
                self.logger.info("Closing ClearML task...")
                self.task.close()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier.")
    parser.add_argument('mode', choices=['train', 'predict'], help='Mode of operation: train or predict')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')

    args = parser.parse_args()

    if args.mode == 'train':
        model_type = input("Enter the model type for training ('random_forest' or 'catboost'): ").strip()
    else:
        model_type = input("Enter the model type for prediction ('random_forest' or 'catboost'): ").strip()

    model = My_Classifier_Model(args.mode, model_type=model_type)

    if args.mode == 'train':
        model.train(args.dataset)
    elif args.mode == 'predict':
        model.predict(args.dataset)


if __name__ == '__main__':
    main()
