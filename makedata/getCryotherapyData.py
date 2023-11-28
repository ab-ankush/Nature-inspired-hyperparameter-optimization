from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import os


def load_data(train_and_test=True):
    """
    loads the cryotherapy dataset
    """

    path = os.path.join("datastore/Cryotherapy.csv")
    df = pd.read_csv(path)
    cols = df.columns[:-1]
    target = df.columns[-1]

    X = df[cols].values
    y = df[target]

    # Encoding the targets
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    if not train_and_test:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    # Normalize the data
    for i in range(X_train.shape[1]):
        std_scaler = StandardScaler()
        X_train[:, i] = std_scaler.fit_transform(X_train[:, i].reshape(-1, 1)).reshape(-1,)
        X_test[:, i] = std_scaler.transform((X_test[:, i].reshape(-1, 1))).reshape(-1,)

    return X_train, X_test, y_train, y_test
