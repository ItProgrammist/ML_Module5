import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(filepath, mode='train'):
    data = pd.read_csv(filepath)

    if mode == 'train':
        X = data.drop(['PassengerId', 'Cabin', 'Destination', 'Name', 'Transported'], axis=1)
        y = data['Transported']
        X = pd.get_dummies(X)
        X = X.fillna(X.median(numeric_only=True))
        X = X.astype(int)
        X = X.drop(['CryoSleep_False', 'VIP_False'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return [X_train, X_test, y_train, y_test]

    elif mode == 'test':
        X_final = data.drop(['PassengerId', 'Cabin', 'Destination', 'Name'], axis=1)
        X_final = pd.get_dummies(X_final)
        X_final = X_final.fillna(X_final.median(numeric_only=True))
        X_final = X_final.astype(int)
        X_final = X_final.drop(['CryoSleep_False', 'VIP_False'], axis=1)
        return [X_final, data]

    else:
        raise ValueError("Invalid mode. Please choose 'train' or 'test'.")
