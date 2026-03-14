import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(
        X,
        y,
        target_col="Class",
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
):

    if train_size + val_size + test_size != 1.0:
        raise ValueError("Train, validation and test sizes must sum to 1")


    # first split → train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    # split temp → val + test
    val_ratio_adjusted = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio_adjusted),
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test