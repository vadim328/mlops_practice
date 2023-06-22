import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import model_preparation


model_preparation.preparation_model()
pkl_filename = "my_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

X_test = np.load("test/X_test.npy")
y_test = pd.read_csv("test/y_test.csv").iloc[:, 1].values
predictions = model.predict(X_test)
print("Model test accuracy is: "+str(round(accuracy_score(predictions, y_test), 3)))
