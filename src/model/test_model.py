import numpy as np
import pandas as pd
import os
import pytest
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
pred = np.load("src/model/output.npy")


def test_inefernce():
    assert type(pred) == type(y_test)


def test_process_data():
    assert len(X) > 0
    

def test_model_slicing():
    with open('src/model/slice_model_output.txt', 'r') as f_r:
        content = f_r.read()
        assert len(content) > 0





#test_inefernce(y_test, pred)


#test_process_data(X)


#test_model_slicing('src/model/slice_model_output.txt')