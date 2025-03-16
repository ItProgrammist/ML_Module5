#!/usr/bin/env python

from preprocessing import preprocess_data
from models import train_random_forest
from sklearn.metrics import accuracy_score
from utils import save_prediction_csv
import pandas as pd
from pathlib import Path


def main():


    filepath = str(Path(__file__).resolve().parent)
    print('Loading and processing data...')
    X_train, X_test, y_train, y_test, X_final, test_data = preprocess_data(filepath)
    print('Fitting Random Forest model with GridSearchCV...')
    best_rf_clf, accuracy = train_random_forest(X_train, X_test, y_train, y_test)

    print(f'The best Random Forest model: {best_rf_clf}')
    print('\nBest params for the model:')
    print(best_rf_clf.get_params())
    print(f'Acuracy on X_test data set: {accuracy:.2f}')
    
    try:
        print('Writing the predicted data (DataFrame) into .csv file...')
        y_pred_all = best_rf_clf.predict(X_final)
        final_prediction_df = pd.DataFrame(data={'PassengerId': test_data.PassengerId, 'Transported': y_pred_all})
        # save_prediction_csv(filepath + '../results_data/prediction.csv', final_prediction_df)
        final_prediction_df.to_csv(filepath + '/../results_data/prediction.csv', index=False)
        print('Written successfully!')
    except:
        print('Error occurred while writing the data.')

if __name__ == '__main__':
    main()