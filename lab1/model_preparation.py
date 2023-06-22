from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import pickle
import data_preprocessing


def preparation_model():
    data_preprocessing.preprocessing_data()
    X_train = np.load("train/X_train.npy")
    y_train = pd.read_csv("train/y_train.csv").iloc[:, 1].values
    model = DecisionTreeClassifier(max_depth=31)
    model.fit(X_train, y_train)

    pkl_filename = "my_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
