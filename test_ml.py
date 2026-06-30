import os
import pandas as pd
from sklearn.model_selection import (train_test_split)
from sklearn.ensemble import RandomForestClassifier
from train_model import cat_features
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


project_path = "."
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)


def test_train_model():
    """
    # Test train_model function
    """
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
        )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference():
    """
    Test that inference returns predictions of the expected shape and type
    """
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
        )
    X_test, _, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
        )
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(X_test)


def test_compute():
    """
    Test compute model metrics returns values in the expected range
    """
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
