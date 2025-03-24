#!/usr/bin/env python

import argparse
import pandas as pd
import joblib
from pathlib import Path
import logging
import traceback
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
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.model_name = f'{model_type}_model.joblib'
        self.logger = Logger().logger
        
        self.setup_clearml()

        self.task = Task.init(
            project_name='model_project', 
            task_name='train_model', 
            task_type=Task.TaskTypes.optimizer
        )

    def setup_clearml(self):
        self.api_server = 'http://<your-server-address>'
        self.web_server = 'http://<your-server-address>'
        self.access_key = '<your-access-key>'
        self.secret_key = '<your-secret-key>'
        
        from clearml import Task
        Task.init(
            project_name='model_project',
            task_name='train_model',
            api_server=self.api_server,
            web_server=self.web_server,
            access_key=self.access_key,
            secret_key=self.secret_key
        )

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

        if self.model_type == 'random_forest':
            self.logger.info('----- Fitting Random Forest model with GridSearchCV... -----')
            print('----- Fitting Random Forest model with GridSearchCV... -----')

            try:
                self.model, accuracy = train_random_forest(X_train, X_test, y_train, y_test)
                self.logger.info(f'The best Random Forest model: {self.model}')
                self.logger.info(f'Best params for the model: {self.model.get_params()}')
                self.logger.info(f'Accuracy on X_test data set: {accuracy:.4f}')
                print(f'The best Random Forest model: {self.model}')
                print(f'Best params for the model:')
                print(self.model.get_params())
                print(f'Accuracy on X_test data set: {accuracy:.4f}')
            except Exception as e:
                self.logger.error(f'Error while training Random Forest: {e}')
                print(f'Error while training Random Forest: {e}')
                return

        elif self.model_type == 'catboost':
            self.logger.info('----- Fitting CatBoost model with GridSearchCV... -----')
            print('----- Fitting CatBoost model with GridSearchCV... -----')

            try:
                self.model, accuracy = train_catboost(X_train, X_test, y_train, y_test)
                self.logger.info(f'The best CatBoost model: {self.model}')
                self.logger.info(f'Best params for the model: {self.model.get_params()}')
                self.logger.info(f'Accuracy on X_test data set: {accuracy:.4f}')
                print(f'The best CatBoost model: {self.model}')
                print(f'Best params for the model:')
                print(self.model.get_params())
                print(f'Accuracy on X_test data set: {accuracy:.4f}')
            except Exception as e:
                self.logger.error(f'Error while training CatBoost: {e}')
                print(f'Error while training CatBoost: {e}')
                return

        self.save_model()

    def save_model(self):
        model_path = Path('./model') / self.model_name
        try:
            joblib.dump(self.model, model_path)
            self.logger.info(f'Model {self.model_name} saved successfully to {model_path}')
            print(f'Model {self.model_name} saved successfully!')
        except Exception as e:
            self.logger.error(f'Error saving model {self.model_name}: {e}')
            print(f'Error saving model {self.model_name}: {e}')

    def predict(self, dataset_filename):
        model_path = Path('./model') / self.model_name
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f'Model loaded successfully from {model_path}')
            print(f'Model loaded successfully from {model_path}')
        except Exception as e:
            self.logger.error(f'Error loading model from {model_path}: {e}')
            print(f'Error loading model from {model_path}: {e}')
            return

        self.logger.info('Loading and processing data for prediction...')
        print('Loading and processing data for prediction...')

        try:
            X_final, test_data = preprocess_data(dataset_filename, mode='test')
            self.logger.info('Data loaded and processed successfully for prediction.')
        except Exception as e:
            error_message = f"Error while processing data: {e}\n{traceback.format_exc()}"
            self.logger.error(error_message)
            print(error_message)
            return

        self.logger.info(f'Making predictions using {self.model_type} model...')
        print(f'Making predictions using {self.model_type} model...')

        predicted_file_name = f'prediction_{self.model_type}.csv'
        self.logger.info(f'{self.model_type.capitalize()}: Writing the predicted data (DataFrame) into .csv file...')
        print(f'{self.model_type.capitalize()}: Writing the predicted data (DataFrame) into .csv file...')
        try:
            y_pred_all = self.model.predict(X_final)
            final_prediction_df = pd.DataFrame(data={'PassengerId': test_data.PassengerId, 'Transported': y_pred_all})
            final_prediction_df.to_csv(f'./results_data/{predicted_file_name}', index=False)
            self.logger.info('Written successfully!')
            print('Written successfully!')
        except Exception as e:
            self.logger.error(f'Error occurred while writing the data for {self.model_type}: {e}')
            print(f'Error occurred while writing the data for {self.model_type}.')

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier.")
    parser.add_argument('mode', choices=['train', 'predict'], help='Mode of operation: train or predict')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')

    args = parser.parse_args()

    if args.mode == 'train':
        model_type = input("Enter the model type for training ('random_forest' or 'catboost'): ").strip()
    else:
        model_type = input("Enter the model type for prediction ('random_forest' or 'catboost'): ").strip()

    model = My_Classifier_Model(model_type=model_type)

    if args.mode == 'train':
        model.train(args.dataset)
    elif args.mode == 'predict':
        model.predict(args.dataset)

if __name__ == '__main__':
    main()
