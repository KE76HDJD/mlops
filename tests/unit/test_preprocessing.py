import pandas as pd


def test_churn_dataset_exists():
    df = pd.read_csv("data/raw/churn.csv")

    assert not df.empty


def test_churn_columns_exist():
    df = pd.read_csv("data/raw/churn.csv")

    expected_columns = [
        "customer_id",
        "tenure",
        "monthly_charges",
        "support_calls",
        "contract",
        "churn"
    ]

    for column in expected_columns:
        assert column in df.columns


def test_churn_target_values():
    df = pd.read_csv("data/raw/churn.csv")

    assert set(df["churn"].unique()).issubset({0, 1})