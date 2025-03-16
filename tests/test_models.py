from src.preprocessing import load_data, preprocess_data
from src.models import train_model

def test_model():
    df = load_data('data/train.csv')
    df = preprocess_data(df)
    model = train_model(df)
    assert model is not None
