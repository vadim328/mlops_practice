import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocessing_data():
    X_train = pd.read_csv("train/X_train.csv")
    X_test = pd.read_csv("test/X_test.csv")
    scaler = MinMaxScaler()
    scaler.fit(X_train) # для тренировочных сначала "обучаем"
    X_train = scaler.transform(X_train) # потом преобразуем  transform
    X_test = scaler.transform(X_test)  # для тестовых - просто transform

    np.save('train/X_train', X_train)
    np.save('test/X_test', X_test)
