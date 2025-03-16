# import joblib

# def save_model(model, filepath):
#     joblib.dump(model, filepath)

# def load_model(filepath):
#     return joblib.load(filepath)

import pandas as pd

def save_prediction_csv(df, filepath):
    df.to_csv(filepath, index=False)