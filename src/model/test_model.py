import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import load

from .preprocess_data import process_data
from .model import inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

model = load("src/model/model.joblib")
encoder = load("src/model/encoder.joblib")
lb = load("src/model/lb.joblib")
df = pd.read_csv('src/data/census_cleaned.csv').drop('Unnamed: 0', axis=1)
X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb, label='salary')
y = df['salary']
y = lb.fit_transform(y.values).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
pred = inference(model, X_test)



def test_inefernce(y_test, y_pred):
    pass
    #assert len(y_pred) == len(y_test)
    #assert type(y_pred) == type(y_test)

def test_process_data(x):
    assert len(x) > 0
    
def test_model_slicing(path):
    with open(path, 'r') as f_r:
        content = f_r.read()
        assert len(content) > 0





test_inefernce(y_test, pred)


test_process_data(X)


test_model_slicing('src/model/slice_model_output.txt')
